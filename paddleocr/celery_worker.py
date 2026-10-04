"""
Celery worker для выполнения OCR задач в фоне.
"""

import os
import sys
import logging
from pathlib import Path
import cv2
import time
from celery import Celery
from celery.signals import worker_process_init, worker_shutdown
import signal

from ocr_service import PaddleOCRService, enhance_image, smart_filter


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ocr-worker")

# Конфигурация
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "/data" if IS_PRODUCTION else "./temp_storage"))
STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# Celery приложение
app = Celery(
    'ocr_tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_time_limit=600,
    task_soft_time_limit=540,
    result_expires=3600,
    task_default_retry_delay=10,
    task_max_retries=3,
)

# Глобальная переменная для модели
ocr_service = None


@worker_process_init.connect
def init_worker(**kwargs):
    """
    Инициализирует PaddleOCR-модель в каждом процессе воркера.

    Вызывается Celery-сигналом worker_process_init при старте каждого
    дочернего процесса. Создает экземпляр PaddleOCRService с ограничением
    максимальной стороны изображения (env MAX_IMAGE_SIDE, по умолчанию 2500)
    и сохраняет его в глобальной переменной ocr_service.

    Args:
        **kwargs: Служебные аргументы Celery (не используются).

    Returns:
        None

    Raises:
        Exception: Если инициализация модели не удалась — воркер падает,
            чтобы Celery мог перезапустить процесс.
    """
    global ocr_service

    logger.info(f"Initializing PaddleOCR model (PID: {os.getpid()})...")

    try:
        max_image_side = int(os.getenv("MAX_IMAGE_SIDE", "2500"))
        ocr_service = PaddleOCRService(max_image_side=max_image_side)

        if ocr_service.ocr is not None:
            logger.info("PaddleOCR model loaded")
        else:
            logger.error("Failed to load PaddleOCR model")

    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise


@worker_shutdown.connect
def shutdown_worker(**kwargs):
    """
    Освобождает ресурсы OCR-сервиса при завершении процесса воркера.

    Вызывается Celery-сигналом worker_shutdown. Если сервис был
    инициализирован — вызывает ocr_service.cleanup() и обнуляет
    глобальную ссылку.

    Args:
        **kwargs: Служебные аргументы Celery (не используются).

    Returns:
        None
    """
    global ocr_service

    if ocr_service:
        logger.info("Cleaning up OCR service...")
        ocr_service.cleanup()
        ocr_service = None


@app.task(
    name='worker.process_ocr',
    bind=True,
    max_retries=3,
    queue='ocr_queue'
)
def process_ocr_task(self, image_path_str: str, whitelist: str, enhance: bool):
    """
    Основная Celery-задача распознавания текста на изображении.

    Пайплайн:
        1. Проверяет существование файла (с короткой задержкой 2 с —
           на случай гонки при записи файла во внешнем хранилище).
        2. Читает изображение через cv2.imread.
        3. Если enhance=True — применяет enhance_image, иначе
           использует оригинал.
        4. Проверяет, что ocr_service инициализирован.
        5. Вызывает ocr_service.recognize(processed_array) и замеряет
           время распознавания.
        6. Применяет smart_filter(raw_text, whitelist).
        7. Формирует словарь-результат с текстом, метаданными и
           идентификатором задачи.
        8. В finally удаляет файл изображения (best-effort).

    При ошибке:
        - Если попыток меньше max_retries — вызывает self.retry(exc=e).
        - Иначе пробрасывает исключение дальше.

    Args:
        self: Экземпляр задачи (bind=True) — дает доступ к self.request
            и self.retry.
        image_path_str (str): Путь к изображению на диске.
        whitelist (str): Строка допустимых символов для OCR.
        enhance (bool): Нужна ли предобработка изображения перед OCR.

    Returns:
        dict: {
            "text": str,                    # отфильтрованный текст
            "raw_text": str,                # сырой вывод OCR
            "engine": "paddleocr",
            "enhanced": bool,
            "whitelist_used": str,
            "processing_time_seconds": float,
            "image_shape": tuple[int, int], # (height, width)
            "task_id": str
        }

    Raises:
        FileNotFoundError: Если файл изображения не найден даже после
            повторной проверки.
        ValueError: Если cv2.imread вернул None (битый/нечитаемый файл).
        RuntimeError: Если ocr_service не инициализирован.
        celery.exceptions.Retry: При повторной попытке выполнения
            (self.retry).
    """
    global ocr_service

    task_id = self.request.id
    logger.info(f"Processing task {task_id}")

    image_path = Path(image_path_str)

    if not image_path.exists():
        time.sleep(2)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image_array = cv2.imread(str(image_path))

        if image_array is None:
            raise ValueError(f"Failed to read image: {image_path}")

        logger.info(f"Image loaded: {image_array.shape}")

        if enhance:
            logger.info("✨ Applying enhancement...")
            processed_array = enhance_image(image_array)
        else:
            processed_array = image_array

        if ocr_service is None:
            raise RuntimeError("OCR service not initialized")

        logger.info("Starting OCR...")
        start_time = time.time()

        raw_text = ocr_service.recognize(processed_array)
        recognition_time = time.time() - start_time

        filtered_text = smart_filter(raw_text, whitelist)

        result = {
            "text": filtered_text,
            "raw_text": raw_text,
            "engine": "paddleocr",
            "enhanced": enhance,
            "whitelist_used": whitelist,
            "processing_time_seconds": round(recognition_time, 2),
            "image_shape": image_array.shape[:2],
            "task_id": task_id
        }

        logger.info(f" Task {task_id} completed in {recognition_time:.2f}s")

        return result

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")

        if self.request.retries < self.max_retries:
            logger.info(f"Retrying (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e)

        raise

    finally:
        try:
            if image_path.exists():
                image_path.unlink()
        except:
            pass


def handle_signal(signum, frame):
    """
    Обработчик сигналов SIGTERM/SIGINT: аккуратно останавливает воркер.

    Вызывает ocr_service.cleanup() (если сервис инициализирован) и
    завершает процесс через sys.exit(0).

    Args:
        signum (int): Номер полученного сигнала.
        frame: Текущий стек-фрейм (не используется).

    Returns:
        None
    """
    logger.info(f"Received signal {signum}, shutting down...")
    if ocr_service:
        ocr_service.cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


if __name__ == '__main__':
    app.start()