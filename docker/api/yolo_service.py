import logging
import os
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# 1. Ограничение количества потоков ONNX Runtime
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
ort.set_default_logger_severity(3)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/app/models/best.onnx"
TARGET_CLASSES = [0, 1, 2, 4, 7]
MAX_DIRECT_SIZE = 1536

app = FastAPI(title="YOLO Detection Service")


# 2. Обычная глобальная функция загрузки моделей
def load_models():
    logger.info("Загрузка моделей в память...")

    yolo_model = YOLO(MODEL_PATH, task="detect")

    # Инициализация SAHI
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model=yolo_model,
        confidence_threshold=0.45,
        device="cpu",
        image_size=1344,
    )
    return yolo_model, sahi_model


model, detection_model = load_models()


def _predict_standard(image: np.ndarray):
    results = model(image, conf=0.5, classes=TARGET_CLASSES, verbose=False)
    return results[0].boxes.data.cpu().numpy().tolist()


def _predict_with_sahi(image: np.ndarray):
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=1536,
        slice_width=1536,
        overlap_height_ratio=0.1,  
        overlap_width_ratio=0.1,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
        verbose=0,
    )

    detections = []
    for pred in result.object_prediction_list:
        if pred.category.id in TARGET_CLASSES:
            bbox = pred.bbox.to_xyxy()
            detections.append(
                [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                    float(pred.score.value),
                    float(pred.category.id),
                ]
            )
    return detections


@app.post("/detect_raw")
async def detect_raw(file: UploadFile = File(...)):
    """

    Распознает элементы на изображении и 
    выбирает метод в зависимости от размера изображения,
    если размер изображения > MAX_DIRECT_SIZE,
    тогда применяется sahi (распознавание по кусочкам)

    Args:
        - file: UploadFile, optional - передаваетое изображение

    Returns:
        - dict: содержит информацию о успешности распознавания,
        о результатах детекции, необходимых для анализа

    Raises:
        - HTTPException
    
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400, detail="Невалидное изображение"
            )

        height, width = image.shape[:2]

        if height <= MAX_DIRECT_SIZE and width <= MAX_DIRECT_SIZE:
            logger.info(f"Standard YOLO mode ({width}x{height})")
            detections = _predict_standard(image)
            mode_used = "standard"
        else:
            logger.info(f"SAHI mode ({width}x{height})")
            detections = _predict_with_sahi(image)
            mode_used = "sahi"

        return {
            "success": True,
            "results": detections,
            "names": model.names,
            "mode": mode_used,
            "num_detections": len(detections),
        }

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "YOLO Detection API is running"}