import requests
from typing import List, Dict
import qf
from urllib.parse import quote
import logging
import os

logger = logging.getLogger(__name__)

class DBClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("DB_API_URL", "http://db-service:8000")
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Accept-Charset': 'utf-8'
        })

    def search_breakers(self, qf_obj: qf.QF) -> list:
        try:
            # Формируем параметры запроса
            params = {
                "ID_QF": qf_obj.ID_QF if qf_obj else '',  # Исправлено
                "Current": qf_obj.Current if qf_obj else '',  # Исправлено
                "Voltage": qf_obj.Voltage if qf_obj else '',  # Исправлено
                "Current_Close": qf_obj.Current_Close if qf_obj else ''  # Исправлено
            }
            
            # Кодируем для URL
            encoded_params = {k: quote(str(v)) for k, v in params.items()}
            
            logger.debug(f"Отправка запроса с параметрами: {params}")
            
            response = self.session.get(
                f"{self.base_url}/breakers",
                params=encoded_params,
                timeout=5
            )
            response.raise_for_status()
            return response.json().get("data", [])
            
        except Exception as e:
            print(f"Ошибка запроса: {str(e)}")
            raise
    def search_counters(self, limit: int = 10) -> List[Dict]:
        response = self.session.get(
            f"{self.base_url}/counters",
            params={"limit": limit},
            timeout=5
            )
        return response.json().get("data", [])
    
    def search_transformators(self, limit: int = 10) -> List[Dict]:
        response = self.session.get(
            f"{self.base_url}/transformators",
            params={"limit": limit},
            timeout=5
        )
        return response.json().get("data", [])
    
