# Используем slim-образ с Python 3.12
FROM python:3.12-slim

# 1. Обновляем индекс пакетов с повтором при ошибке

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && ln -s /usr/lib/x86_64-linux-gnu/libgthread-2.0.so.0 /usr/lib/ \
    && rm -rf /var/lib/apt/lists/*

# 2. Установка Python-зависимостей с зеркалом PyPI (если нужно)
RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    urllib3 \
    python-multipart \
    requests \
    python-dotenv \
    fastapi \
    gunicorn \
    uvicorn \
    pydantic

# 3. Создание структуры проекта
RUN useradd -m -u 1000 appuser

# 4. Копирование файлов (оптимизированный порядок)
COPY --chown=appuser:appuser ./src/qf.py ./src/ta.py ./src/search_text.py ./src/search_object.py src/circuit_graph.py /app/
COPY --chown=appuser:appuser ./db_api/db_client_api.py /app/
COPY --chown=appuser:appuser ./front_api/front_server_api.py /app/

# 5. Настройка окружения
WORKDIR /app
USER appuser

# 6. Запуск приложения
# Запуск приложения с оптимизированными параметрами Gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
"front_server_api:app",\ 
    "--bind", "0.0.0.0:8002",\
    "--workers", "4",   \
    "--threads", "2", \                        
    "--timeout", "300",  \                       
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--worker-tmp-dir", "/dev/shm", \      
    "--max-requests", "1000",  \           
    "--max-requests-jitter", "50",  \
    "--log-level", "info", \
    "--access-logfile", "-", \
    "--error-logfile", "-",    \
    "--capture-output",      \
    "--graceful-timeout", "30"   \
]