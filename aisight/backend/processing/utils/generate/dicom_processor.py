import pydicom
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple, Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)


class DICOMProcessor:
    def __init__(self, target_size: Tuple[int, int, int] = (128, 128, 128)):
        self.target_size = target_size
        self.logger = logger

    def load_dicom_series(self, folder_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Загрузка и обработка серии DICOM файлов. Возвращает (volume, metadata) или None при ошибке."""
        try:
            # 1. Сбор и валидация DICOM-файлов
            dicom_datasets = []
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if not os.path.isfile(file_path):
                    continue
                try:
                    # Читаем только метаданные для быстрой проверки
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    # Дополнительная проверка: должен быть медицинский модальностью (CT, MR и т.д.)
                    if hasattr(ds, 'Modality') and ds.Modality in ['CT', 'MR', 'PT', 'XA', 'CR', 'DX']:
                        dicom_datasets.append((file_path, ds))
                except Exception:
                    continue  # Пропускаем не-DICOM файлы

            if not dicom_datasets:
                self.logger.warning(f"No valid DICOM files found in {folder_path}")
                return None

            # Сортируем по пути к файлу, чтобы потом читать пиксели в правильном порядке
            dicom_datasets.sort(key=lambda x: x[1].get('InstanceNumber', 0))

            # 2. Проверка, что все срезы из одной серии
            first_ds_meta = dicom_datasets[0][1]
            series_uid = getattr(first_ds_meta, 'SeriesInstanceUID', None)
            if series_uid:
                dicom_datasets = [
                    (fp, ds) for fp, ds in dicom_datasets
                    if getattr(ds, 'SeriesInstanceUID', None) == series_uid
                ]

            # 3. Полная загрузка пикселей и сортировка по позиции среза
            full_datasets = []
            for file_path, _ in dicom_datasets:
                try:
                    ds = pydicom.dcmread(file_path)  # Теперь читаем полностью
                    full_datasets.append(ds)
                except Exception as e:
                    self.logger.warning(f"Failed to read pixel data from {file_path}: {e}")
                    continue

            if not full_datasets:
                self.logger.error(f"No DICOM files with pixel data in {folder_path}")
                return None

            # Сортировка по физической позиции среза
            full_datasets.sort(key=self._get_slice_position)

            # 4. Проверка совместимости срезов
            ref_shape = full_datasets[0].pixel_array.shape
            for ds in full_datasets:
                if ds.pixel_array.shape != ref_shape:
                    self.logger.error(f"Inconsistent slice shapes in {folder_path}: {ref_shape} vs {ds.pixel_array.shape}")
                    return None

            # 5. Извлечение метаданных
            metadata = self._extract_metadata(full_datasets[0])

            # 6. Создание 3D объема
            volume = self._create_volume(full_datasets)
            if volume.ndim != 3:
                self.logger.error(f"Volume is not 3D after stacking: shape {volume.shape}")
                return None
            
            if volume.shape[0] < 3 or volume.shape[1] < 10 or volume.shape[2] < 10:
                self.logger.error(f"Volume too small: {volume.shape}")
                return None

            # 7. Препроцессинг
            volume = self._preprocess_volume(volume, metadata)


            return volume, metadata

        except Exception as e:
            self.logger.error(f"Unexpected error loading DICOM series from {folder_path}: {e}")
            return None

    def _get_slice_position(self, ds) -> float:
        """Извлечение позиции среза по ImagePositionPatient или SliceLocation."""
        try:
            if hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient is not None:
                return float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation') and ds.SliceLocation is not None:
                return float(ds.SliceLocation)
        except (IndexError, ValueError, TypeError):
            pass
        return 0.0

    def _extract_metadata(self, ds) -> Dict:
        """Извлечение ключевых метаданных."""
        return {
            'study_uid': str(getattr(ds, 'StudyInstanceUID', '')),
            'series_uid': str(getattr(ds, 'SeriesInstanceUID', '')),
            'patient_id': str(getattr(ds, 'PatientID', '')),
            'study_date': str(getattr(ds, 'StudyDate', '')),
            'modality': str(getattr(ds, 'Modality', '')),
            'pixel_spacing': list(getattr(ds, 'PixelSpacing', [1.0, 1.0])),
            'slice_thickness': float(getattr(ds, 'SliceThickness', 1.0)),
            'rows': int(getattr(ds, 'Rows', ref_shape[0] if (ref_shape := getattr(ds, 'pixel_array', np.zeros((1,1)))).shape else 512)),
            'columns': int(getattr(ds, 'Columns', ref_shape[1] if (ref_shape := getattr(ds, 'pixel_array', np.zeros((1,1)))).shape else 512)),
        }

    def _create_volume(self, dicom_files: List) -> np.ndarray:
        """Создание 3D объема из DICOM файлов."""
        slices = []
        for ds in dicom_files:
            try:
                pixel_array = ds.pixel_array.astype(np.float32)
                # Применение RescaleSlope и RescaleIntercept (обязательно для CT)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                slices.append(pixel_array)
            except Exception as e:
                self.logger.warning(f"Error processing pixel array: {e}")
                return np.array([])  # вызовет ошибку выше

        if not slices:
            raise ValueError("No valid slices after pixel processing")


        return np.stack(slices, axis=0)  # Shape: (D, H, W)

    def _preprocess_volume(self, volume: np.ndarray, metadata: Dict) -> np.ndarray:
        """Препроцессинг: HU clipping, нормализация, ресайз."""
        modality = metadata.get('modality', '').upper()
        
        
        # Для КТ — HU нормализация
        if modality == 'CT':
            volume = np.clip(volume, -1000, 1000)
            volume = (volume + 1000) / 2000.0  # [0, 1]
        else:
            # Для MR/других — min-max нормализация
            p5, p95 = np.percentile(volume, [5, 95])
            volume = np.clip(volume, p5, p95)
            volume = (volume - p5) / (p95 - p5 + 1e-8)

        # Ресайз к целевому размеру
        volume = self._resize_volume(volume, self.target_size)
        
        # ДОБАВЬТЕ ЭТУ СТРОЧКУ: добавляем channel dimension для MONAI
        volume = volume[np.newaxis, ...]  # (1, 128, 128, 128)
        
        return volume

    def _resize_volume(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Изменение размера 3D тома с помощью trilinear интерполяции."""
        
        
        # ПРОВЕРКА: volume должен быть 3D
        if volume.ndim != 3:
            print(f"WARNING: Expected 3D volume, got {volume.ndim}D")
            # Если volume 2D, преобразуем в 3D
            if volume.ndim == 2:
                volume = volume[np.newaxis, :, :]  # Добавляем Z-ось
                print(f"Converted 2D to 3D: {volume.shape}")
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Тензор должен быть: (batch, channels, D, H, W)
            vol_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
            print(f"Tensor shape before interpolation: {vol_tensor.shape}")
            
            resized = F.interpolate(
                vol_tensor, 
                size=target_shape, 
                mode='trilinear', 
                align_corners=False
            )
            
            result = resized.squeeze().numpy().astype(np.float32)
            
            return result
            
        except ImportError:
            # Альтернативная реализация с scipy
            from scipy.ndimage import zoom
            zoom_factors = (
                target_shape[0] / volume.shape[0],
                target_shape[1] / volume.shape[1], 
                target_shape[2] / volume.shape[2]
            )
            result = zoom(volume, zoom_factors, order=1)
            print(f"Scipy result shape: {result.shape}")
            return result