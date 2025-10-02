# AISight

**Краткое описание**

AISight — это распределённое веб-приложение для задач компьютерного зрения с серверной частью на Python (FastAPI) и фронтендом на JavaScript (Vue). Проект предназначен для быстрого прототипирования сервисов, принимающих изображения/видео, запускающих ML-пайплайны и возвращающих результаты в виде файлов (например, Excel-таблицы с результатами анализа).

> В репозитории присутствуют две основные части: `backend` (FastAPI) и `frontend` (Vue). Запуск возможен локально и через контейнеры.

---

## Основные возможности

* API `/predict` принимает ZIP-архив с изображениями, запускает пайплайн анализа (`processing.scripts.inference.main`) и формирует файл с результатами `result.xlsx`.
* API `/processing/extract/output/result.xlsx` позволяет скачать готовый результат.
* Веб-интерфейс (Vue) для загрузки изображений и визуального контроля работы.
* Простая структура проекта для добавления собственных ML-моделей и сервисов.

## Ограничения

* В репозитории нет встроенных обученных моделей (вызов `processing.scripts.inference.main` предполагает реализацию в стороннем модуле). Перед использованием в боевой среде необходимо добавить/обучить модели и проверить корректность пайплайна.
* Производительность и масштабируемость зависят от конфигурации оборудования и выбранных моделей.

---

## Системные требования

Минимальные:

* Python 3.10+ (рекомендуется 3.11)
* Node.js 16+ и npm
* Git
* Для запуска с Docker: Docker & docker-compose
* Рекомендуется 8+ ГБ RAM (меньше — возможно, но медленнее)

---

## Быстрый старт (Quick Start)

1. **Клонировать репозиторий**

```bash
git clone https://github.com/Alyukin/AISight.git
cd AISight
```

2. **Запустить backend (локально)**

```bash
cd backend
python3 -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend по умолчанию будет слушать `http://127.0.0.1:8000`.

3. **Запустить frontend (локально)**

```bash
cd ../frontend
npm install
npm run dev
```

Откройте в браузере адрес, который выдаст dev-сервер (обычно `http://localhost:5173`). Фронтенд должен общаться с бэкендом по API.

4. **Пример использования API (curl)**

```bash
# Отправить ZIP-файл на анализ
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "zip_file=@/path/to/archive.zip"

# Скачать результат
curl -O "http://127.0.0.1:8000/processing/extract/output/result.xlsx"
```

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
├─ Info.txt
└─ README.md               # этот файл
```

**Описание ключевых файлов**

* `backend/main.py` — точка старта FastAPI; определяет эндпоинты:

  * `GET /` — проверка работоспособности API.
  * `POST /predict` — принимает ZIP с изображениями, запускает inference и возвращает путь к файлу `result.xlsx`.
  * `GET /processing/extract/output/result.xlsx` — отдаёт результат обработки.
* `backend/requirements.txt` — список Python-зависимостей.
* `frontend/package.json` — npm-скрипты и зависимости фронтенда.
* `Info.txt` — дополнительная информация по проекту (если имеется).

---

## Руководство по развертыванию

### 1) Локальное развёртывание (development)

Следуйте шагам из раздела «Быстрый старт». Для удобства запуска можно использовать две отдельные терминальные сессии — для backend и frontend.

### 2) Производственное развёртывание (production)

Так как в проекте нет Dockerfile, можно использовать следующие шаблоны.

#### Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/ ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile

```dockerfile
FROM node:18 as build
WORKDIR /app
COPY frontend/ ./
RUN npm install && npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build: 
      context: .
      dockerfile: ./backend/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend/processing:/app/processing

  frontend:
    build: 
      context: .
      dockerfile: ./frontend/Dockerfile
    ports:
      - "5173:80"
```

---

## Руководство пользователя

### Общие принципы

1. Перейдите в веб-интерфейс (обычно `http://localhost:5173`).
2. Загрузите ZIP-файл с изображениями через форму загрузки.
3. Дождитесь обработки и скачайте `result.xlsx` с результатами анализа.

### Примеры API-запросов

* Проверить статус API:

```bash
curl http://127.0.0.1:8000/
```

* Запрос на анализ ZIP-файла:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "zip_file=@test_data.zip"
```

* Скачать Excel с результатами:

```bash
curl -O "http://127.0.0.1:8000/processing/extract/output/result.xlsx"
```

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
