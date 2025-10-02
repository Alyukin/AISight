# AISight

Веб-приложение для анализа медицинских изображений формата DICOM с FastAPI бэкендом и Vue фронтендом.

Сервис запускает ML-пайплайн (MONAI) и возвращает результаты анализа в виде Excel-таблицы
> Запуск возможен локально и через контейнеры.

## Системные требования
Минимальные:
* Python 3.10+ (рекомендуется 3.10)
* Node.js 16+ и npm
* Git
* Для запуска с Docker: Docker & docker-compose

## Быстрый старт (Quick Start)

```bash
git clone https://github.com/Alyukin/AISight.git
cd AISight
```
- #### Важно!
    Необходимо скачать model.pth из (https://github.com/Alyukin/AISight/releases/download/Download_model/model.pth) и поместить в /backend/processing/extract/model

**Запустить Backend (локально)**  

```bash
cd backend
python3 -m venv venv
source venv/bin/activate # Linux/macOS
# .\venv\Scripts\Activate.ps1 # Windows
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```
Фронтенд общается с бэкендом по API.
SwaggerUI (FastAPI) -  http://127.0.0.1:8000/docs

**Запустить Frontend (локально)**
```bash
cd ../frontend
npm install

npm run dev
```

Приложение будет доступно по адресу http://localhost:5173

---

## Структура проекта

```
AISight/
├─ backend/                # серверная часть (FastAPI)
│  ├─ main.py              # входная точка приложения
│  ├─ requirements.txt     # зависимости Python
│  └─ processing/          # пайплайны обработки и inference
├─ frontend/               # веб-интерфейс (Vue)
│  ├─ package.json
│  ├─ src/
│  └─ ...
├─ .gitattributes
├─ .gitignore
└─ README.md               # этот файл
```

**Описание ключевых файлов**

* `backend/main.py` — точка старта FastAPI; определение эндпоинтов.
* `backend/requirements.txt` — список Python-зависимостей.
* `frontend/package.json` — npm-скрипты и зависимости фронтенда.

**API Endpoints**

* `GET /` - {"message": "AI Sight API is running"} - Проверка работы API
* `POST /predict` - Принимает `zip_file`, запускает пайплайн анализа (`processing.scripts.inference.main`), формирует файл с результатами `result.xlsx`, возвращает JSON с путём к результату.
* `GET /processing/extract/output/result.xlsx` - Позволяет скачать результаты анализа в Excel.

**Пример использования API (curl)**

```bash
# Отправить ZIP-файл на анализ
curl -X POST "http://127.0.0.1:8000/predict" \
-F "zip_file=@/path/to/archive.zip"

# Скачать результат
curl -O "http://127.0.0.1:8000/processing/extract/output/result.xlsx"
```

---
## Руководство по развёртыванию

### Docker (рекомендуется)

В проекте уже есть Dockerfile для backend и frontend, а также `docker-compose.yml`.

Запуск:

```bash
docker-compose up --build
```

Backend будет доступен на `http://localhost:8000`, Frontend — на `http://localhost:5173`.

### Production-рекомендации

* Использовать **nginx** как обратный прокси перед backend и для отдачи статических файлов frontend.
* Настроить переменные окружения (например, `CORS`, пути сохранения файлов, параметры моделей).
* Запускать контейнеры через `docker-compose -d` или в Kubernetes.

### Общие принципы

1. Перейдите в веб-интерфейс (обычно `http://localhost:5173`).
2. Загрузите ZIP-файл с изображениями через форму загрузки.
3. Дождитесь обработки и скачайте `result.xlsx` с результатами анализа.



### Советы по отладке

* Если фронтенд не показывает результаты — проверьте в браузере консоль и вкладку Network.
* Посмотрите логи backend (uvicorn) — там будут ошибки обработки.
* Убедитесь, что CORS разрешает фронтенду доступ к backend (`http://localhost:5173`).

---

## Контрибьюция

Если хотите помочь проекту:

1. Форкните репозиторий и создайте ветку feature/my-feature.
2. Сделайте изменения и добавьте тесты/инструкции.
3. Откройте Pull Request с описанием изменений.

---
