import os
import sys
import ctypes
from front.menu import main
from db_api.db_server_api import app as db_app
from front_api.front_server_api import app as front_app
import uvicorn
import threading
import pytesseract
import requests
import locale
import base64
import time
from typing import Optional

class APIClient:
    def __init__(self, base_url="http://localhost:8002"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def detect_image(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(
                    f"{self.base_url}/detect_images/",
                    files=files,
                    timeout=600
                )
            
            if response.status_code != 200:
                raise Exception(f"Ошибка API ({response.status_code}): {response.text}")
            
            data = response.json()
            # Декодируем base64 обратно в байты
            image_bytes = base64.b64decode(data["image_base64"])
            return image_bytes, data["detection_results"]
        except Exception as e:
            raise Exception(f"Ошибка при обращении к API: {str(e)}")

def setup_tesseract():
    """Настраивает путь к Tesseract в .exe и разработке."""
    if getattr(sys, 'frozen', False):
        tesseract_path = os.path.join(os.path.dirname(sys.executable), "Tesseract-OCR", "tesseract.exe")
    else:
        tesseract_path = "C:/Program Files/Tesseract-OCR/tesseract.exe" 
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(tesseract_path), "tessdata")

def is_server_running(port: int) -> bool:
    """Проверяет, запущен ли сервер на указанном порту"""
    try:
        return requests.get(
            f"http://localhost:{port}/health",
            timeout=2
        ).status_code == 200
    except:
        return False

# def run_db_api():
#     """Запускает сервер БД на порту 8001"""
#     if not is_server_running(8001):
#         ctypes.windll.kernel32.AllocConsole()
#         sys.stdout = open('CONOUT$', 'w')
#         sys.stderr = open('CONOUT$', 'w')
#         sys.stdin = open('CONIN$', 'r')
#         config = uvicorn.Config(
#             db_app,
#             host="0.0.0.0",
#             port=8001,
#             log_level="info"
#         )
#         server = uvicorn.Server(config)
#         server.run()

def run_front_api():
    """Запускает фронтенд API на порту 8002"""
    if not is_server_running(8002):
        ctypes.windll.kernel32.AllocConsole()
        sys.stdout = open('CONOUT$', 'w')
        sys.stderr = open('CONOUT$', 'w')
        sys.stdin = open('CONIN$', 'r')
        config = uvicorn.Config(
            front_app,
            host="0.0.0.0",
            port=8002,
            log_level="info"
        )
        server = uvicorn.Server(config)
        server.run()

def wait_for_server(port: int, timeout: int = 10) -> bool:
    """Ожидает запуска сервера"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_server_running(port):
            return True
        time.sleep(0.5)
    return False

def initialize_apis() -> Optional[APIClient]:
    """Инициализирует API серверы и возвращает клиент"""
    try:
        # # Запуск сервера БД
        # db_thread = threading.Thread(target=run_db_api, daemon=True)
        # db_thread.start()
        
        # Запуск фронтенд API
        front_thread = threading.Thread(target=run_front_api, daemon=True)
        front_thread.start()
        

        if not wait_for_server(8002):
            raise Exception("Не удалось запустить API серверы")
        # # Ожидаем запуска серверов
        # if not wait_for_server(8001) or not wait_for_server(8002):
        #     raise Exception("Не удалось запустить API серверы")
        
        return APIClient()
    except Exception as e:
        print(f"Ошибка инициализации API: {str(e)}")
        return None

if __name__ == "__main__":
    # Настройка окружения
    #setup_tesseract()
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') 
    
    # Инициализация API
    api_client = initialize_apis()
    
    if api_client is None:
        print("Не удалось инициализировать API")
    
    # Запуск главного приложения с передачей клиента API
    main(api_client=api_client)