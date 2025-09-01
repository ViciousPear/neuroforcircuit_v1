# Используем slim-образ с Python 3.12
FROM python:3.12-slim

# 1. Обновляем индекс пакетов с повтором при ошибке
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Установка Python-зависимостей с зеркалом PyPI (если нужно)
RUN pip install --no-cache-dir --retries 5 \
    psycopg2-binary \
    python-dotenv \
    apscheduler \
    xmltodict \
    requests

# 3. Создание структуры проекта
RUN useradd -m -u 1000 appuser && \
     mkdir -p /app/db_connect \
     mkdir -p /app/database

# 4. Копирование файлов
COPY --chown=appuser:appuser ./database/connection.py /app/
COPY --chown=appuser:appuser ./database/parsing.py /app/
COPY --chown=appuser:appuser ./database/.env /app/database
COPY --chown=appuser:appuser ./database/db_connect/.env /app/db_connect

# 5. Настройка окружения
WORKDIR /app
USER appuser

# 6. Запуск приложения
CMD ["python", "parsing.py"]