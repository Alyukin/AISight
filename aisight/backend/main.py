from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import pydicom
import torch
from monai.transforms import Compose, AddChannel, ScaleIntensity, ToTensor
from monai.networks.nets import UNet  # пример UNet, замените на вашу модель

app = FastAPI()

# Разрешаем запросы с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # или ["http://localhost:5173"]
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

# Пример UNet, замените на свою обученную модель
model = UNet(
    dimensions=2,
    in_channels=1,
    out_channels=2,  # норм/патология
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2)
).to(device)

# Если есть чекпоинт:
# model.load_state_dict(torch.load("model_checkpoint.pth", map_location=device))
model.eval()

# Трансформации MONAI для инференса
transform = Compose([
    AddChannel(),
    ScaleIntensity(),
    ToTensor()
])

# -------------------------------
# Endpoint для предсказания
# -------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(dicom: UploadFile = File(...)):
    # 1. Читаем DICOM
    contents = await dicom.read()
    dicom_file = BytesIO(contents)
    ds = pydicom.dcmread(dicom_file)
    pixel_array = ds.pixel_array.astype(np.float32)

    # 2. Подготовка данных для MONAI
    img_tensor = transform(pixel_array).unsqueeze(0).to(device)  # [1,1,H,W]

    # 3. Инференс
    with torch.no_grad():
        output = model(img_tensor)  # [1,2,H,W]
        probs = torch.softmax(output, dim=1)
        conf, pred_class = torch.max(probs, dim=1)

    # 4. Генерация heatmap (канал "патология")
    heatmap_array = probs[0, 1].cpu().numpy() * 255
    heatmap_array = np.uint8(heatmap_array)
    heatmap_img = Image.fromarray(heatmap_array).convert("RGBA")

    # Красим в красный с прозрачностью
    for y in range(heatmap_img.height):
        for x in range(heatmap_img.width):
            r, g, b, a = heatmap_img.getpixel((x, y))
            heatmap_img.putpixel((x, y), (255, 0, 0, int(a * 0.3)))

    buffered = BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()

    return PredictionResponse(
        prediction="pathology" if pred_class.item() == 1 else "normal",
        confidence=float(conf.item()),
        heatmap=f"data:image/png;base64,{heatmap_base64}"
    )
