from fastapi import FastAPI,  HTTPException, Query
from fastapi.responses import RedirectResponse
from typing import Dict, List
from pydantic import BaseModel
import qf
import database_api
import logging
from urllib.parse import unquote


app = FastAPI(
    title="Internal DB API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
    )
logger = logging.getLogger(__name__)


class BreakerItem(BaseModel):
    id: str
    name: str
    price: float

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/breakers", response_model=Dict[str, List[BreakerItem]])
async def get_breakers(
    ID_QF: str = Query(""),
    Current: str = Query(""),
    Voltage: str = Query(""),
    Current_Close: str = Query("")
) -> Dict[str, List[dict]]:
    try:
        # Создаем объект QF через Pydantic
        new_qf = qf.QF(
        ID_QF= unquote(ID_QF),
        Current= unquote(Current),
        Voltage= unquote(Voltage),
        Current_Close= unquote(Current_Close)
        )
        
        print("Получены параметры:", new_qf)

        # Ваша бизнес-логика
        result = database_api.DatabaseAPI.search_qf_breakers(new_qf)
        return {
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки: {str(e)}"
        )
@app.get("/counters")
async def get_counters(limit: int = 10):
    result = database_api.DatabaseAPI.search_counters(limit)
    return {
            "data": result
        }

@app.get("/transformators")  
async def get_transformators(limit: int = 10):
    result = database_api.DatabaseAPI.search_transformators(limit)
    return {
            "data": result
        }

# if __name__ == "__main__":
    
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# перевод из двоичного кода в норм изображения(надо ее переделать чутка)
# нада сделать перевод в двоичный код чтобы передавать изображения(вспомогательную)