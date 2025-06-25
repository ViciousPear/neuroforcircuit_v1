from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from typing import Dict, List
from src.search_object import detect_one_image
from pydantic import BaseModel
import logging
import tempfile
import os
import cv2
import base64
from ultralytics import YOLO

app = FastAPI(
    title="FRONT API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Инициализация модели YOLO
model_path = "runs/restudying_neuro_v5.71s/weights/best.pt"
model = YOLO(model_path)

logger = logging.getLogger(__name__)

class DetectionResponse(BaseModel):
    """Модель ответа для API"""
    image_base64: str  # Изображение в формате base64
    detection_results: List  # Результаты детекции

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/detect_images/", response_model=DetectionResponse)
async def detect_images(file: UploadFile = File(...)):
    try:
        # Проверка 1: Файл передан?
        if not file:
            raise HTTPException(status_code=400, detail="Файл не был передан")

        # Проверка 2: Это изображение? (если content_type отсутствует, проверяем расширение)
        filename = file.filename.lower()
        if not (filename.endswith(('.png', '.jpg', '.jpeg', '.bmp'))):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением (.png, .jpg, .jpeg, .bmp)")

        # Читаем файл
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Передан пустой файл")

        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Обработка изображения
        results_list = detect_one_image(temp_file_path, model)
        
        # Конвертируем в base64
        _, img_encoded = cv2.imencode('.png', results_list[0])
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        return DetectionResponse(
            image_base64=img_base64,
            detection_results=results_list[1]
        )

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)