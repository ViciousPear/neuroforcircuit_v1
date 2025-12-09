from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import logging

app = FastAPI(title="YOLO Detection Service")
logger = logging.getLogger(__name__)

# Загрузка модели при старте
model = YOLO("/app/models/best.pt")
# model = YOLO("runs/restudying_neuro_v6.5s/weights/best.pt")

@app.post("/detect_raw")
async def detect_raw(file: UploadFile = File(...)):
    """Только детекция объектов без постобработки"""
    try:
        # Чтение изображения
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Выполнение детекции
        results = model(image, conf=0.5, classes=[0,1,2,4,8])
        
        # Возвращаем сырые результаты
        return {
            "success": True,
            "results": results[0].boxes.data.cpu().numpy().tolist(),  # Координаты и классы
            "names": model.names  # Названия классов
        }
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))