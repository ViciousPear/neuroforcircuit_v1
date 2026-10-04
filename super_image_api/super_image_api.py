# super_image_api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
import base64
from PIL import Image
import logging
import sys
import traceback
from typing import Optional
# from src import super_image_processor - для локального использования
from super_image_processor import SuperImageFixed


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="Super-Image Enhancement API",
    description="API для улучшения качества изображений с помощью нейронных сетей",
    version="1.0.0"
)

# Глобальный объект процессора
image_processor = None


@app.on_event("startup")
async def startup_event():
    """
    Инициализирует глобальный процессор Super-Image при старте приложения.

    Создает экземпляр SuperImageFixed(scale=2) и сохраняет его в
    глобальной переменной image_processor. При ошибке инициализации
    логирует сообщение и traceback, но не прерывает запуск приложения —
    сервис остается в состоянии "unhealthy".

    Returns:
        None
    """
    global image_processor
    try:
        logger.info("Инициализация Super-Image процессора...")
        image_processor = SuperImageFixed(scale=2)
        logger.info("Super-Image процессор инициализирован успешно")
    except Exception as e:
        logger.error(f"Ошибка инициализации процессора: {str(e)}")
        logger.error(traceback.format_exc())


@app.get("/")
async def root():
    """
    Корневой эндпоинт: возвращает информацию о сервисе.

    Returns:
        dict: {
            "service": str,
            "version": str,
            "status": "running" | "error",
            "endpoints": dict[str, str]
        }
    """
    return {
        "service": "Super-Image Enhancement API",
        "version": "1.0.0",
        "status": "running" if image_processor else "error",
        "endpoints": {
            "/enhance": "POST - улучшение изображения",
            "/health": "GET - проверка здоровья сервиса",
            "/info": "GET - информация о модели"
        }
    }


@app.get("/health")
async def health_check():
    """
    Проверка здоровья сервиса.

    Returns:
        tuple | dict: При успехе — {"status": "healthy", "model_loaded": True}.
        При неудаче — ({"status": "unhealthy", "model_loaded": False}, 503).
    """
    if image_processor:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}, 503


@app.get("/info")
async def model_info():
    """
    Возвращает информацию о загруженной модели.

    Returns:
        dict: {
            "model_name": str,
            "scale": int,
            "device": "cpu",
            "status": "ready"
        }

    Raises:
        HTTPException(503): Если модель не загружена.
    """
    if not image_processor:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    return {
        "model_name": image_processor.model_name,
        "scale": image_processor.scale,
        "device": "cpu",
        "status": "ready"
    }


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """
    Конвертирует bytes в изображение OpenCV (BGR).

    Сначала пытается декодировать через cv2.imdecode. Если результат None
    (например, формат не поддержан OpenCV), пробует открыть через PIL и
    сконвертировать RGB -> BGR.

    Args:
        image_bytes (bytes): Байты изображения (PNG, JPEG и т.п.).

    Returns:
        np.ndarray: Изображение OpenCV BGR.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Пробуем как RGB
        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return img


def cv2_to_bytes(img: np.ndarray, format: str = "PNG") -> bytes:
    """
    Конвертирует изображение OpenCV в bytes заданного формата.

    Args:
        img (np.ndarray): Изображение OpenCV.
        format (str): Формат кодирования ("PNG", "JPG", "JPEG").
            По умолчанию "PNG".

    Returns:
        bytes: Байты закодированного изображения.

    Raises:
        ValueError: Если cv2.imencode не смог закодировать изображение.
    """
    success, encoded_image = cv2.imencode(f".{format.lower()}", img)
    if not success:
        raise ValueError(f"Не удалось закодировать изображение в формат {format}")
    return encoded_image.tobytes()


@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    use_tiles: bool = True,
    scale: Optional[int] = 2,
    return_base64: bool = True,
    format: str = "png"
):
    """
    Улучшает качество изображения через Super-Image модель.

    Пайплайн:
        1. Проверяет, что модель загружена (иначе 503).
        2. Проверяет допустимый формат (png/jpg/jpeg).
        3. Читает файл, декодирует в OpenCV BGR (bytes_to_cv2).
        4. Если scale отличается от текущего — логирует намерение
           изменить масштаб (фактическое изменение не реализовано).
        5. Вызывает image_processor.enhance(original_image, use_tiles).
        6. Формирует JSON-ответ с размерами, scale_factor,
           processing_time="N/A" и (опционально) base64 изображения.

    Args:
        file (UploadFile): Загружаемое изображение.
        use_tiles (bool): Использовать ли тайловую обработку
            (рекомендуется для больших изображений). По умолчанию True.
        scale (int | None): Коэффициент увеличения (2 или 4). По умолчанию 2.
        return_base64 (bool): Возвращать ли изображение в base64.
            По умолчанию True.
        format (str): Формат выходного изображения ("png", "jpg",
            "jpeg"). По умолчанию "png".

    Returns:
        JSONResponse: {
            "success": True,
            "original_size": {"height", "width", "channels"},
            "enhanced_size": {"height", "width", "channels"},
            "scale_factor": int,
            "processing_time": "N/A",
            "image_base64": str (если return_base64=True),
            "image_format": str (если return_base64=True),
            "image_size_bytes": int (если return_base64=True)
        }

    Raises:
        HTTPException(503): Модель не загружена.
        HTTPException(400): Пустой файл, неподдерживаемый формат или
            не удалось декодировать изображение.
        HTTPException(500): Ошибка обработки изображения.
    """
    if not image_processor:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Сервис не готов."
        )

    # Проверяем поддерживаемый формат
    allowed_formats = ["png", "jpg", "jpeg"]
    if format.lower() not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат. Допустимые: {allowed_formats}"
        )

    try:
        # Чтение изображения
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")

        logger.info(f"Получено изображение: {file.filename}, размер: {len(contents)} bytes")

        # Конвертируем в OpenCV формат
        original_image = bytes_to_cv2(contents)

        if original_image is None or original_image.size == 0:
            raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")

        logger.info(f"Размер оригинального изображения: {original_image.shape}")

        # Изменяем масштаб процессора если нужно
        if scale != image_processor.scale:
            logger.info(f"Изменение масштаба с {image_processor.scale} на {scale}")

        # Улучшение изображения
        logger.info(f"Начало улучшения (use_tiles={use_tiles})...")
        enhanced_image = image_processor.enhance(original_image, use_tiles=use_tiles)

        if enhanced_image is None:
            raise HTTPException(status_code=500, detail="Ошибка обработки изображения")

        logger.info(f"Размер улучшенного изображения: {enhanced_image.shape}")

        # Подготовка ответа
        response_data = {
            "success": True,
            "original_size": {
                "height": original_image.shape[0],
                "width": original_image.shape[1],
                "channels": original_image.shape[2] if len(original_image.shape) > 2 else 1
            },
            "enhanced_size": {
                "height": enhanced_image.shape[0],
                "width": enhanced_image.shape[1],
                "channels": enhanced_image.shape[2] if len(enhanced_image.shape) > 2 else 1
            },
            "scale_factor": image_processor.scale,
            "processing_time": "N/A"
        }

        # Если нужно вернуть изображение
        if return_base64:
            enhanced_bytes = cv2_to_bytes(enhanced_image, format)
            img_base64 = base64.b64encode(enhanced_bytes).decode('utf-8')

            response_data["image_base64"] = img_base64
            response_data["image_format"] = format
            response_data["image_size_bytes"] = len(enhanced_bytes)

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.post("/enhance_roi")
async def enhance_roi(
    file: UploadFile = File(...),
    x1: int = 0,
    y1: int = 0,
    x2: int = 100,
    y2: int = 100,
    use_tiles: bool = False,
    return_base64: bool = True
):
    """
    Улучшает конкретную область изображения (ROI).

    Пайплайн:
        1. Проверяет, что модель загружена (иначе 503).
        2. Читает файл, декодирует в OpenCV BGR.
        3. Проверяет корректность координат ROI относительно размеров
           изображения.
        4. Вырезает ROI, прогоняет через image_processor.enhance.
        5. Формирует JSON-ответ с координатами ROI, размерами
           оригинала и результата и (опционально) base64 улучшенного ROI.

    Args:
        file (UploadFile): Загружаемое изображение.
        x1 (int): Левая граница ROI. По умолчанию 0.
        y1 (int): Верхняя граница ROI. По умолчанию 0.
        x2 (int): Правая граница ROI. По умолчанию 100.
        y2 (int): Нижняя граница ROI. По умолчанию 100.
        use_tiles (bool): Использовать ли тайловую обработку.
            По умолчанию False.
        return_base64 (bool): Возвращать ли ROI в base64. По умолчанию True.

    Returns:
        JSONResponse: {
            "success": True,
            "roi_coordinates": {"x1", "y1", "x2", "y2"},
            "original_roi_size": {"height", "width"},
            "enhanced_roi_size": {"height", "width"},
            "image_base64": str (если return_base64=True),
            "image_format": "png" (если return_base64=True)
        }

    Raises:
        HTTPException(503): Модель не загружена.
        HTTPException(400): Некорректные координаты ROI.
        HTTPException(500): Ошибка обработки ROI.
    """
    if not image_processor:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        # Чтение изображения
        contents = await file.read()
        original_image = bytes_to_cv2(contents)

        # Проверка координат
        h, w = original_image.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            raise HTTPException(
                status_code=400,
                detail=f"Некорректные координаты ROI. Изображение: {w}x{h}, ROI: ({x1},{y1})-({x2},{y2})"
            )

        roi = original_image[y1:y2, x1:x2]
        logger.info(f"Вырезан ROI: {roi.shape}")

        enhanced_roi = image_processor.enhance(roi, use_tiles=use_tiles)

        if enhanced_roi is None:
            raise HTTPException(status_code=500, detail="Ошибка обработки ROI")

        # Подготовка ответа
        response_data = {
            "success": True,
            "roi_coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "original_roi_size": {"height": y2 - y1, "width": x2 - x1},
            "enhanced_roi_size": {
                "height": enhanced_roi.shape[0],
                "width": enhanced_roi.shape[1]
            }
        }

        if return_base64:
            enhanced_bytes = cv2_to_bytes(enhanced_roi, "png")
            img_base64 = base64.b64encode(enhanced_bytes).decode('utf-8')
            response_data["image_base64"] = img_base64
            response_data["image_format"] = "png"

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки ROI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)