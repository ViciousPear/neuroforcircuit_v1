import os
import sys
import ctypes
from front.menu import main
from db_api.db_server_api import app  
import uvicorn
import threading
import pytesseract
import requests
import locale
import io


def setup_tesseract():
    """Настраивает путь к Tesseract в .exe и разработке."""
    if getattr(sys, 'frozen', False):
        # Путь к tesseract.exe в распакованных файлах .exe
        tesseract_path = os.path.join(os.path.dirname(sys.executable), "Tesseract-OCR", "tesseract.exe")
    else:
        # Путь для разработки (если Tesseract в PATH)
        tesseract_path = "C:/Program Files/Tesseract-OCR/tesseract.exe" 
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(tesseract_path), "tessdata")

def is_server_running(port=8001):
    """Проверяет, запущен ли уже сервер на указанном порту"""
    try:
        return requests.get(f"http://localhost:{port}/").status_code == 200
    except:
        return False

def run_api():
    if not is_server_running():
        ctypes.windll.kernel32.AllocConsole()
        sys.stdout = open('CONOUT$', 'w')
        sys.stderr = open('CONOUT$', 'w')
        sys.stdin = open('CONIN$', 'r')
        config = uvicorn.Config(app, host="0.0.0.0", port=8001)
        server = uvicorn.Server(config)
        server.run()



        
if __name__ == "__main__":

    setup_tesseract()
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') 
    if not is_server_running():
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
    main()
    

    # Запуск GUI
