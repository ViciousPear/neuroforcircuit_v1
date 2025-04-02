from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI()


# перевод из двоичного кода в норм изображения(надо ее переделать чутка)
@app.post("/predict")
async def predict(file: UploadFile = File('C:/nekitlox/NeuroForCircuit/not_based.png')):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Здесь вызываем ваш существующий код
    from src.search_object import detect_one_image
    results = detect_one_image(img)
    
    return JSONResponse(results)

# нада сделать перевод в двоичный код чтобы передавать изображения(вспомогательную)