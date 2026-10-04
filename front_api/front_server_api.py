from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from typing import List, Dict, Any
from pydantic import BaseModel
import logging
import tempfile
import os
import cv2
import base64
# from src import search_object - для локального теста
import search_object


app = FastAPI(
    title="FRONT API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    """
    Корневой эндпоинт: перенаправляет на страницу документации.

    Returns:
        RedirectResponse: Редирект на /docs.
    """
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """
    Проверка здоровья сервиса.

    Returns:
        dict: {"status": "OK"}.
    """
    return {"status": "OK"}


class DetectionOnlyResponse(BaseModel):
    """
    Модель ответа API детекции.

    Attributes:
        image_base64 (str): Изображение с нарисованными bbox в формате
            base64 (JPEG).
        detected_elements (List[Dict[str, Any]]): Список распознанных
            элементов схемы. Каждый элемент — словарь с полями
            "id", "type", "number", "confidence", "parameters"
            (см. search_object.recognize_only).
    """
    image_base64: str  # Изображение в формате base64
    detected_elements: List[Dict[str, Any]]  # Результаты детекции


@app.post("/detect_only/", response_model=DetectionOnlyResponse)
async def detect_only(file: UploadFile = File(...)):
    """
    Выполняет детекцию схемы по загруженному изображению БЕЗ поиска в БД.

    Пайплайн:
        1. Проверяет, что файл передан и его расширение входит в
           допустимые (.png, .jpg, .jpeg, .bmp).
        2. Читает содержимое файла и проверяет, что оно не пустое.
        3. Сохраняет файл во временный .jpg на диске.
        4. Вызывает search_object.recognize_only(temp_file_path),
           который прогоняет изображение через YOLO + text-service и
           возвращает (image, detected_elements).
        5. Приводит detected_elements к списку словарей: dict-элементы
           остаются как есть, объекты с __dict__ разворачиваются,
           остальные оборачиваются в {"raw": ..., "index": i}.
        6. Кодирует изображение в JPEG -> base64.
        7. Возвращает DetectionOnlyResponse с изображением и списком
           элементов.
        8. В finally удаляет временный файл (best-effort).

    Args:
        file (UploadFile): Загружаемое изображение схемы.
            Допустимые расширения: .png, .jpg, .jpeg, .bmp.

    Returns:
        DetectionOnlyResponse: {
            "image_base64": str,
            "detected_elements": list[dict]
        }

    Raises:
        HTTPException(400): Файл не передан, имеет недопустимое
            расширение или пустой.
        HTTPException(501): Функция recognize_only отсутствует в
            search_object (AttributeError).
        HTTPException(500): Прочие ошибки распознавания
            (в т.ч. если detected_elements не список).
    """
    temp_file_path = None
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Файл не был передан")

        filename = file.filename.lower()
        if not (filename.endswith(('.png', '.jpg', '.jpeg', '.bmp'))):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Передан пустой файл")

        # Сохранение во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Вызов функции recognize_only
        image, detected_elements = search_object.recognize_only(temp_file_path)

        if image is None:
            raise HTTPException(
                status_code=500,
                detail="Не удалось получить изображение после распознавания"
            )

        if not isinstance(detected_elements, list):
            raise HTTPException(
                status_code=500,
                detail=f"detected_elements должен быть списком, получен: {type(detected_elements)}"
            )

        processed_elements = []
        for i, elem in enumerate(detected_elements):
            if isinstance(elem, dict):
                processed_elements.append(elem)
            else:
                try:
                    if hasattr(elem, '__dict__'):
                        processed_elements.append(elem.__dict__)
                    else:
                        processed_elements.append({"raw": str(elem), "index": i})
                except Exception as e:
                    processed_elements.append({"error": str(e), "index": i})

        _, img_encoded = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        logger.info(f"Распознано элементов: {len(processed_elements)}")
        for elem in processed_elements:
            logger.info(f"Элемент: {type(elem)} - {elem}")

        return DetectionOnlyResponse(
            image_base64=img_base64,
            detected_elements=processed_elements
        )

    except AttributeError as e:
        logger.error(f"Функция recognize_only не существует: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=501,
            detail=f"Функция recognize_only не реализована: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Ошибка распознавания: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка распознавания: {str(e)}"
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл: {str(e)}")