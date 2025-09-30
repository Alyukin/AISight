from zip_dicom_to_3d import ZipDicomTo3D
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import pydicom
import torch

# MONAI импорт
from monai.transforms import Compose, ScaleIntensity, ToTensor
from monai.networks.nets import UNet

app = FastAPI()

# Разрешаем запросы с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # пока для теста, потом заменить на адрес фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель ответа
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    heatmap: str  # base64 PNG
    error_message: str = ""

# -------------------------------
# Инициализация модели MONAI
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

model.eval()  # режим инференса

# Простой пайплайн трансформаций
transform = Compose([
    ScaleIntensity(),
    ToTensor()
])

# -------------------------------
# Вспомогательные функции
# -------------------------------
def create_heatmap(heatmap_array, original_shape):
    """Создание heatmap с наложением на оригинальное изображение"""
    heatmap_normalized = (heatmap_array - heatmap_array.min()) / (heatmap_array.max() - heatmap_array.min() + 1e-8)

    # Создаем цветовую карту
    heatmap_colored = np.zeros((*heatmap_array.shape, 3), dtype=np.uint8)
    heatmap_colored[..., 0] = (heatmap_normalized * 255).astype(np.uint8)

    alpha = (heatmap_normalized * 0.5 * 255).astype(np.uint8)

    heatmap_rgba = np.zeros((*heatmap_array.shape, 4), dtype=np.uint8)
    heatmap_rgba[..., 0] = heatmap_colored[..., 0]
    heatmap_rgba[..., 3] = alpha

    if heatmap_array.shape != original_shape:
        heatmap_img = Image.fromarray(heatmap_rgba).resize(original_shape[::-1], Image.Resampling.LANCZOS)
    else:
        heatmap_img = Image.fromarray(heatmap_rgba)

    return heatmap_img

# -------------------------------
# Endpoint для предсказания
# -------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(dicom: UploadFile = File(...)):
    try:
        # 1. Читаем DICOM
        contents = await dicom.read()
        dicom_file = BytesIO(contents)
        ds = pydicom.dcmread(dicom_file)
        pixel_array = ds.pixel_array.astype(np.float32)
        original_shape = pixel_array.shape

        print("DICOM shape:", pixel_array.shape) # Проверка 
        print("DICOM dtype:", pixel_array.dtype)

        # 2. Добавляем канал вручную (C, H, W)
        pixel_array = np.expand_dims(pixel_array, axis=0)

        # 3. Применяем трансформации
        transformed = transform(pixel_array)
        img_tensor = transformed.unsqueeze(0).to(device)  # [B, C, H, W]

        # 4. Инференс (если модель пока пустая — используем заглушку)
        with torch.no_grad():
            output = model(img_tensor)  # [1,2,H,W]
            probs = torch.softmax(output, dim=1)  # [1,2,H,W]

            # conf_map = максимум вероятности по классам для каждого пикселя: shape [1,H,W]
            conf_map, pred_map = torch.max(probs, dim=1)  # conf_map: [1,H,W], pred_map: [1,H,W]

            # 4. Скалярные агрегаты (без .item() на карте)
            confidence = float(conf_map.mean().item())      # скаляр: средняя уверенность по изображению
            pathology_mean = float(probs[0, 1].mean().item())  # средняя вероятность класса "патология"

            # 5. Решение о метке на уровне изображения
            # Вариант: threshold по средней вероятности (0.5 по умолчанию).
            # Можно заменить 0.5 на значение, подобранное на валидации.
            pred_label = "pathology" if pathology_mean > 0.5 else "normal"


        print("probs shape:", probs.shape)           # должно быть [1, 2, 512, 512]
        print("conf_map shape:", conf_map.shape)     # должно быть [1, 512, 512]
        print("confidence value:", confidence)       # одно число

        # 5. Генерация heatmap (канал "патология")
        heatmap_array = probs[0, 1].cpu().numpy()
        heatmap_img = create_heatmap(heatmap_array, original_shape)

        # 7. Сохраняем в base64
        buffered = BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()

        return PredictionResponse(
            prediction=pred_label,
            confidence=confidence,
            heatmap=f"data:image/png;base64,{heatmap_base64}"
        )

    except Exception as e:
        return PredictionResponse(
            prediction="error",
            confidence=0.0,
            heatmap="",
            error_message=str(e)
        )

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/")
async def root():
    return {"message": "AI Sight API is running"}

@app.get("/model-info")
async def model_info():
    return {
        "model_type": "UNet",
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 2,
        "device": str(device)
    }
