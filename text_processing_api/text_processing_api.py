from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import logging
from fastapi.responses import RedirectResponse
import processing_text

#from processing_text search_ta, search_qf

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Processing Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class TextRecognitionRequest(BaseModel):
    image_base64: str  # Всё изображение
    bbox: List[int]    # [x1, y1, x2, y2] координаты текстовой области
    confidence: float
    class_id: int

class ProcessedObject(BaseModel):
    type: str  # "circuit_breaker", "current_transformer", "unknown", "error"
    object: Optional[Any] = None
    original_text: Optional[str] = None
    bbox: Optional[List[int]] = None
    confidence: Optional[float] = None
    error_message: Optional[str] = None

class TextChunk(BaseModel):
    text: str  # Теперь передаем уже распознанный текст!
    bbox: List[int]
    confidence: float
    class_id: int

class BatchRequest(BaseModel):
    chunks: List[TextChunk]  # Список текстовых чанков
    image_size: Optional[List[int]] = None  # [width, height] (опционально)
    image_shape: Optional[List[int]] = None  # [height, width, channels] (опционально)

@app.post("/api/process-text-batch", response_model=List[ProcessedObject])
async def process_text_batch(request: BatchRequest):
    """
    Обрабатывает уже распознанный текст (без вызова Tesseract)
    """
    if not request.chunks:
        return []
    
    results = []
    
    try:
        for chunk in request.chunks:
            text_content = chunk.text
            
            if not text_content:
                results.append({
                    "type": "unknown",
                    "original_text": "",
                    "bbox": chunk.bbox,
                    "confidence": chunk.confidence
                })
                continue
            
            # Ищем QF
            qf_data = processing_text.search_qf(text_content)
            if qf_data:
                results.append({
                    "type": "circuit_breaker",
                    "object": qf_data.__dict__,
                    "original_text": text_content,
                    "bbox": chunk.bbox,
                    "confidence": chunk.confidence
                })
                continue
            
            # Ищем TA
            ta_data = processing_text.search_ta(text_content)
            if ta_data:
                results.append({
                    "type": "current_transformer",
                    "object": ta_data.__dict__,
                    "original_text": text_content,
                    "bbox": chunk.bbox,
                    "confidence": chunk.confidence
                })
                continue
            
            # Неизвестный текст
            results.append({
                "type": "unknown",
                "original_text": text_content,
                "bbox": chunk.bbox,
                "confidence": chunk.confidence
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")