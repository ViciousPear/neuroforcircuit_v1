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
# from src import super_image_processor
from super_image_processor import SuperImageFixed
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создаем FastAPI приложение
app = FastAPI(
    title="Super-Image Enhancement API",
    description="API для улучшения качества изображений с помощью нейронных сетей",
    version="1.0.0"
)

# Глобальный объект процессора
image_processor = None

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global image_processor
    try:
        logger.info("🚀 Инициализация Super-Image процессора...")
        image_processor = SuperImageFixed(scale=2)
        logger.info("✅ Super-Image процессор инициализирован успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации процессора: {str(e)}")
        logger.error(traceback.format_exc())

@app.get("/")
async def root():
    """Информация о сервисе"""
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
    """Проверка здоровья сервиса"""
    if image_processor:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}, 503

@app.get("/info")
async def model_info():
    """Информация о загруженной модели"""
    if not image_processor:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        "model_name": image_processor.model_name,
        "scale": image_processor.scale,
        "device": "cpu",
        "status": "ready"
    }

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Конвертирует bytes в OpenCV изображение"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Пробуем как RGB
        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return img

def cv2_to_bytes(img: np.ndarray, format: str = "PNG") -> bytes:
    """Конвертирует OpenCV изображение в bytes"""
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
    Улучшение качества изображения
    
    Параметры:
    - file: Загружаемое изображение
    - use_tiles: Использовать ли обработку по тайлам (рекомендуется для больших изображений)
    - scale: Коэффициент увеличения (2 или 4)
    - return_base64: Возвращать ли изображение в base64
    - format: Формат выходного изображения (png, jpg)
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
        
        logger.info(f"📥 Получено изображение: {file.filename}, размер: {len(contents)} bytes")
        
        # Конвертируем в OpenCV формат
        original_image = bytes_to_cv2(contents)
        
        if original_image is None or original_image.size == 0:
            raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")
        
        logger.info(f"🖼️ Размер оригинального изображения: {original_image.shape}")
        
        # Изменяем масштаб процессора если нужно
        if scale != image_processor.scale:
            logger.info(f"Изменение масштаба с {image_processor.scale} на {scale}")
            # Здесь можно переинициализировать модель с новым scale
            # Но для простоты используем текущий
        
        # Улучшение изображения
        logger.info(f"🔧 Начало улучшения (use_tiles={use_tiles})...")
        enhanced_image = image_processor.enhance(original_image, use_tiles=use_tiles)
        
        if enhanced_image is None:
            raise HTTPException(status_code=500, detail="Ошибка обработки изображения")
        
        logger.info(f"✅ Размер улучшенного изображения: {enhanced_image.shape}")
        
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
            "processing_time": "N/A"  # Можно добавить тайминг
        }
        
        # Если нужно вернуть изображение
        if return_base64:
            # Конвертируем в bytes
            enhanced_bytes = cv2_to_bytes(enhanced_image, format)
            
            # Кодируем в base64
            img_base64 = base64.b64encode(enhanced_bytes).decode('utf-8')
            
            response_data["image_base64"] = img_base64
            response_data["image_format"] = format
            response_data["image_size_bytes"] = len(enhanced_bytes)
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {str(e)}")
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
    Улучшение конкретной области изображения (ROI)
    
    Параметры:
    - file: Загружаемое изображение
    - x1, y1: Левая верхняя координата ROI
    - x2, y2: Правая нижняя координата ROI
    - use_tiles: Использовать ли обработку по тайлам
    - return_base64: Возвращать ли изображение в base64
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
        
        # Вырезаем ROI
        roi = original_image[y1:y2, x1:x2]
        logger.info(f"Вырезан ROI: {roi.shape}")
        
        # Улучшаем ROI
        enhanced_roi = image_processor.enhance(roi, use_tiles=use_tiles)
        
        if enhanced_roi is None:
            raise HTTPException(status_code=500, detail="Ошибка обработки ROI")
        
        # Подготовка ответа
        response_data = {
            "success": True,
            "roi_coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "original_roi_size": {"height": y2-y1, "width": x2-x1},
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