from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import zipfile
import os
import tempfile
import shutil
from processing.scripts.inference import main

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Разрешаем запросы с любых источников (можно указать конкретные адреса фронтендов)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Папка для обработки
os.makedirs("processing/extract/input", exist_ok=True)
os.makedirs("processing/extract/output", exist_ok=True)

# Функция для извлечения ZIP файла
def extract_zip(zip_file: UploadFile):
    extract_folder = "processing/extract/input/study"
    os.makedirs(extract_folder, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False) as temp_zip:
        contents = zip_file.file.read()
        temp_zip.write(contents)
        temp_zip_path = temp_zip.name

    # Разархивируем ZIP файл
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    os.remove(temp_zip_path)
    return extract_folder

@app.post("/predict")
async def predict(zip_file: UploadFile = File(...)):
    try:
        extract_folder = extract_zip(zip_file)
        main()
        xlsx_file_path = "processing/extract/output/list2.xlsx"
        return {"file_path": f"/processing/extract/output/result.xlsx"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/processing/extract/output/result.xlsx")
async def get_xlsx_file():
    shutil.rmtree("./processing/extract/input/study/")
    return FileResponse(
        "processing/extract/output/result.xlsx", 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
        filename="result.xlsx"
    )

@app.get("/")
async def root():
    return {"message": "AI Sight API is running"}
