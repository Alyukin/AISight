# AISight

**Краткое описание**

AISight — это распределённое веб-приложение для задач компьютерного зрения с серверной частью на Python (FastAPI) и фронтендом на JavaScript (Vue). Проект предназначен для быстрого прототипирования сервисов, принимающих изображения/видео, запускающих ML-пайплайны и возвращающих результаты в виде файлов (например, Excel-таблицы с результатами анализа).

---

## Основные возможности


* API /predict принимает ZIP-архив с изображениями, запускает пайплайн анализа (processing.scripts.inference.main) и формирует файл с результатами result.xlsx.
* API /processing/extract/output/result.xlsx позволяет скачать готовый результат.
* Веб-интерфейс (Vue) для загрузки изображений и визуального контроля работы.
* Простая структура проекта для добавления собственных ML-моделей и сервисов.

---

## Ограничения

* Масштабируемость и производительность зависят от аппаратного обеспечения и сложности пайплайна.

---

## Системные требования

* Python 3.10+ (Рекомендуется 3.10)
* Node.js 20+
* npm
* Docker и docker-compose (для контейнерного развёртывания)
* Рекомендуется 8+ ГБ RAM

---

## Быстрый старт (Quick Start)

### 1. Запуск без Docker

```bash
git clone https://github.com/Alyukin/AISight.git
cd AISight
```

**Backend:**

```bash
cd backend
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# .\venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**

```bash
cd ../frontend
npm install
npm run dev
```

Frontend будет доступен на `http://localhost:5173` и подключаться к backend (`http://localhost:8000`).

### 2. Запуск с Docker Compose

```bash
docker-compose up --build
```

* Backend будет доступен на `http://localhost:8000`
* Frontend — на `http://localhost:5173`

---

## Примеры использования API

Отправить ZIP‑файл на анализ:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "zip_file=@/path/to/archive.zip"
```

Скачать результат:

```bash
curl -O "http://127.0.0.1:8000/processing/extract/output/result.xlsx"
```

---

## Структура проекта

```
AISight/
├─ backend/                # серверная часть (FastAPI)
│  ├─ main.py              # входная точка API
│  ├─ requirements.txt     # зависимости Python
│  └─ processing/          # пайплайны обработки и inference
├─ frontend/               # веб-интерфейс (Vue)
│  ├─ package.json
│  ├─ src/
│  └─ dist/                # сборка фронтенда
├─ docker-compose.yml      # запуск через Docker Compose
├─ backend/Dockerfile      # Dockerfile для backend
├─ frontend/Dockerfile     # Dockerfile для frontend
└─ README.md               # этот файл
```

**Эндпоинты backend:**

* `GET /` — проверка API.
* `POST /predict` — принимает `zip_file`, запускает анализ, возвращает JSON с путём к результату.
* `GET /processing/extract/output/result.xlsx` — отдаёт готовый Excel‑файл.

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

---

## Руководство пользователя

1. Перейдите в браузере на `http://localhost:5173`.
2. Загрузите ZIP‑архив с изображениями.
3. Дождитесь обработки.
4. Скачайте Excel‑файл `result.xlsx` с результатами анализа.

---

## Контрибьюция

1. Форкните репозиторий и создайте ветку `feature/my-feature`.
2. Внесите изменения, добавьте тесты.
3. Откройте Pull Request.

---

## API документация (сводка)

* `GET /` → {"message": "AI Sight API is running"}
* `POST /predict` → принимает `zip_file`, возвращает JSON с путём к результату.
* `GET /processing/extract/output/result.xlsx` → отдаёт Excel‑файл.

---
