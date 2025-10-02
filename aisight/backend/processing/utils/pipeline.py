import torch
import numpy as np
import pandas as pd
import time
import os
from typing import List, Dict, Any
import logging
from .generate.dicom_processor import DICOMProcessor
from .generate.anomaly_detector import CTAnomalyDetector
from .generate.report_generator import ReportGenerator

class CTClassificationPipeline:
    """Основной пайплайн классификации КТ исследований"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logging()
        
        # Инициализация компонентов
        self.dicom_processor = DICOMProcessor(
            target_size=config.get('target_size', (128, 128, 128))
        )
        self.model = self._load_model()
        self.report_generator = ReportGenerator()
        
        self.logger.info(f"Pipeline initialized on device: {self.device}")
    
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_model(self) -> CTAnomalyDetector:
        """Загрузка модели"""
        model = CTAnomalyDetector()
        
        if 'model_path' in self.config and os.path.exists(self.config['model_path']):
            checkpoint = torch.load(self.config['model_path'], map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Model loaded from {self.config['model_path']}")
        else:
            self.logger.warning("No model path provided, using untrained model")
        
        model.to(self.device)
        model.eval()
        return model
    
    def process_study(self, study_path: str) -> Dict[str, Any]:
        """Обработка одного исследования"""
        start_time = time.time()
        result = {
            'path_to_study': study_path,
            'study_uid': '',
            'series_uid': '',
            'probability_of_pathology': 0.0,
            'pathology': 0,
            'processing_status': 'Failure',
            'time_of_processing': 0.0,
            'most_dangerous_pathology_type': 'None',
            'pathology_localization': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'error_message': ''
        }
        
        try:
            # Загрузка и препроцессинг DICOM
            volume, metadata = self.dicom_processor.load_dicom_series(study_path)
            result['study_uid'] = metadata['study_uid']
            result['series_uid'] = metadata['series_uid']
            
            # Подготовка данных для модели
            input_tensor = self._prepare_input(volume)
            
            # Инференс
            with torch.no_grad():
                anomaly_score, reconstructed, bbox = self.model(input_tensor)
            
            # Постобработка результатов
            probability = anomaly_score.item()
            result['probability_of_pathology'] = probability
            result['pathology'] = 1 if probability > self.config.get('threshold', 0.5) else 0
            
            # Определение типа патологии
            if result['pathology'] == 1:
                pathology_type = self._classify_pathology_type(volume, bbox)
                result['most_dangerous_pathology_type'] = pathology_type
                result['pathology_localization'] = self._process_localization(bbox)
            
            result['processing_status'] = 'Success'
            
        except Exception as e:
            error_msg = str(e)
            result['error_message'] = error_msg
            result['processing_status'] = f'Failure: {error_msg}'
            self.logger.error(f"Error processing {study_path}: {error_msg}")
        
        result['time_of_processing'] = time.time() - start_time
        return result
    
    def _prepare_input(self, volume: np.ndarray) -> torch.Tensor:
        """Подготовка входных данных для модели"""
        print(f"Volume shape: {volume.shape}")  # [1, 128, 128, 128]
        
        # volume уже имеет форму [1, 128, 128, 128] где:
        # - 1: вероятно channel dimension
        # - 128, 128, 128: depth, height, width
        
        volume_tensor = torch.tensor(volume).float()
        
        # Проверяем текущую форму
        if len(volume_tensor.shape) == 4:
            # Форма: [channels, depth, height, width] -> нужно [batch, channels, depth, height, width]
            volume_tensor = volume_tensor.unsqueeze(0)  # Добавляем только batch dimension -> [1, 1, 128, 128, 128]
        else:
            raise ValueError(f"Unexpected volume shape: {volume.shape}")
        
        print(f"Final tensor shape: {volume_tensor.shape}")  # Должно быть [1, 1, 128, 128, 128]
        return volume_tensor.to(self.device)
    
    def _classify_pathology_type(self, volume: np.ndarray, bbox: torch.Tensor) -> str:
        """Классификация типа патологии"""
        # Простая эвристическая классификация на основе HU значений
        lung_hu_range = (-1000, -400)
        bone_hu_range = (300, 2000)
        soft_tissue_hu_range = (-100, 300)
        
        # Анализ областей с аномалиями
        bbox_coords = bbox.squeeze().cpu().numpy()
        x_min, x_max, y_min, y_max, z_min, z_max = self._denormalize_bbox(bbox_coords, volume.shape)
        
        roi = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        mean_hu = np.mean(roi) * 2000 - 1000  # Денормализация
        
        if lung_hu_range[0] <= mean_hu <= lung_hu_range[1]:
            return "Pulmonary pathology"
        elif bone_hu_range[0] <= mean_hu <= bone_hu_range[1]:
            return "Bone pathology"
        elif soft_tissue_hu_range[0] <= mean_hu <= soft_tissue_hu_range[1]:
            return "Soft tissue pathology"
        else:
            return "Unknown pathology"
    
    def _process_localization(self, bbox: torch.Tensor) -> List[float]:
        """Обработка локализации патологии"""
        bbox_coords = bbox.squeeze().cpu().numpy()
        return bbox_coords.tolist()
    
    def _denormalize_bbox(self, bbox: np.ndarray, volume_shape: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
        """Денормализация bounding box к оригинальным размерам"""
        z, y, x = volume_shape
        x_min = int(bbox[0] * x)
        x_max = int(bbox[1] * x)
        y_min = int(bbox[2] * y)
        y_max = int(bbox[3] * y)
        z_min = int(bbox[4] * z)
        z_max = int(bbox[5] * z)
        
        return (x_min, x_max, y_min, y_max, z_min, z_max)
    
    def batch_process(self, studies_list: List[str]) -> pd.DataFrame:
        """Пакетная обработка исследований"""
        results = []
        
        for i, study_path in enumerate(studies_list):
            self.logger.info(f"Processing study {i+1}/{len(studies_list)}: {study_path}")
            
            result = self.process_study(study_path)
            results.append(result)
            
            # Проверка времени обработки
            if result['time_of_processing'] > 600:  # 10 минут
                self.logger.warning(f"Study {study_path} processing time exceeded 10 minutes")
        
        # Генерация отчета
        df = self.report_generator.generate_report(results, self.config['output_path'])
        return df