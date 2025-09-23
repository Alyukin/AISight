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
from monai.transforms import Compose, ScaleIntensity, ToTensor, EnsureChannelFirst
from monai.networks.nets import UNet

app = FastAPI()

# Разрешаем запросы с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель ответа
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    heatmap: str  # base64 PNG

# -------------------------------
# Инициализация модели MONAI
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Правильная инициализация UNet для MONAI 1.5.1
model = UNet(
    spatial_dims=2,  # используем spatial_dims вместо dimensions
    in_channels=1,
    out_channels=2,  # норм/патология
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2  # рекомендуется добавить для лучшей производительности
).to(device)

# Если есть чекпоинт:
# model.load_state_dict(torch.load("model_checkpoint.pth", map_location=device))
model.eval()

# Трансформации MONAI для инференса
transform = Compose([
    EnsureChannelFirst(channel_dim=None),
    ScaleIntensity(),
    ToTensor()
])

# -------------------------------
# Вспомогательные функции
# -------------------------------
def create_heatmap(heatmap_array, original_shape):
    """Создание heatmap с наложением на оригинальное изображение"""
    # Нормализуем массив heatmap
    heatmap_normalized = (heatmap_array - heatmap_array.min()) / (heatmap_array.max() - heatmap_array.min() + 1e-8)
    
    # Создаем цветовую карту (красный канал)
    heatmap_colored = np.zeros((*heatmap_array.shape, 3), dtype=np.uint8)
    heatmap_colored[..., 0] = (heatmap_normalized * 255).astype(np.uint8)  # Красный канал
    
    # Создаем альфа-канал (прозрачность)
    alpha = (heatmap_normalized * 0.5 * 255).astype(np.uint8)  # Прозрачность 50%
    
    # Создаем RGBA изображение
    heatmap_rgba = np.zeros((*heatmap_array.shape, 4), dtype=np.uint8)
    heatmap_rgba[..., 0] = heatmap_colored[..., 0]  # R
    heatmap_rgba[..., 3] = alpha  # A
    
    # Изменяем размер до оригинального, если нужно
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

        # 2. Подготовка данных для MONAI
        transformed = transform(pixel_array)
        img_tensor = transformed.unsqueeze(0).to(device)  # [1,1,H,W]

        # 3. Инференс
        with torch.no_grad():
            output = model(img_tensor)  # [1,2,H,W]
            probs = torch.softmax(output, dim=1)
            conf, pred_class = torch.max(probs, dim=1)

        # 4. Генерация heatmap (канал "патология")
        heatmap_array = probs[0, 1].cpu().numpy()
        
        # Создаем heatmap
        heatmap_img = create_heatmap(heatmap_array, original_shape)

        # Сохраняем в base64
        buffered = BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()

        return PredictionResponse(
            prediction="pathology" if pred_class.item() == 1 else "normal",
            confidence=float(conf.mean().item()),  # средняя уверенность по всем пикселям
            heatmap=f"data:image/png;base64,{heatmap_base64}"
        )
    
    except Exception as e:
        return PredictionResponse(
            prediction="error",
            confidence=0.0,
            heatmap="",
            error_message=str(e)
        )

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "AI Sight API is running"}

# Endpoint для проверки модели
@app.get("/model-info")
async def model_info():
    return {
        "model_type": "UNet",
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 2,
        "device": str(device)
    }