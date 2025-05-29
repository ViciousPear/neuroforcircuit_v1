import psycopg2 
from dotenv import load_dotenv
import logging
from datetime import datetime
# dotenv_path = 'database/db_connect/.env.utf8'
# load_dotenv(dotenv_path)

# def setup_ssl_certificate():
#     """Скачивает и устанавливает SSL-сертификат для PostgreSQL"""
#     try:
#         # Создаем папку .postgresql в домашнем каталоге
#         postgresql_dir = Path.home() / ".postgresql"
#         postgresql_dir.mkdir(exist_ok=True)
        
#         # Путь к сертификату
#         cert_path = postgresql_dir / "root.crt"
        
#         # Если сертификат уже существует - пропускаем загрузку
#         if cert_path.exists():
#             print(f"SSL сертификат уже существует: {cert_path}")
#             return str(cert_path)
        
#         # Скачиваем сертификат с помощью curl
#         subprocess.run(
#             ["curl.exe", "-o", str(cert_path), "https://beget.com/cloud-ca.crt"],
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
        
#         print(f"SSL сертификат успешно загружен: {cert_path}")
#         return str(cert_path)
        
#     except Exception as e:
#         print(f"Ошибка при загрузке SSL сертификата: {e}")
#         raise


# def download_certificate():
#     cert_url = "https://beget.com/cloud-ca.crt"
#     cert_dir = Path.home() / ".postgresql"
#     cert_path = cert_dir / "root.crt"
    
#     # Создаём папку, если не существует
#     cert_dir.mkdir(exist_ok=True)
    
#     # Принудительно загружаем свежий сертификат
#     try:
#         urllib.request.urlretrieve(cert_url, cert_path)
#         print(f"✅ Сертификат успешно загружен в {cert_path}")
#         return cert_path
#     except Exception as e:
#         print(f"❌ Ошибка загрузки сертификата: {e}")
#         return None
    
# def get_short_path(long_path):
#     """Конвертирует путь в формат 8.3 (без кириллицы)"""
#     buf = ctypes.create_unicode_buffer(500)
#     ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 500)
#     return buf.value

# def verify_certificate(cert_path):
#     """Проверяет доступность и читаемость сертификата"""
#     try:
#         if not os.path.exists(cert_path):
#             print(f"Файл не существует: {cert_path}")
#             return False
        
#         with open(cert_path, 'rb') as f:
#             content = f.read()
#             if b"BEGIN CERTIFICATE" not in content:
#                 print("Файл не содержит валидный сертификат")
#                 return False
            
#         print("Сертификат валиден")
#         return True
        
#     except Exception as e:
#         print(f"Ошибка проверки сертификата: {e}")
#         return False

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('postgres_connections.log'),  # Логи в файл
        logging.StreamHandler()  # Логи в консоль
    ]
)
logger = logging.getLogger('PostgresConnector')

def connect_to_postgres():
    """
    Подключается к PostgreSQL (neuro_base).
    """
    start_time = datetime.now()
    try:
        logger.info("Попытка подключения к PostgreSQL...")
        logger.debug(f"Параметры подключения: host=82.202.129.245, port=5432, dbname=neuro_base")

        connection = psycopg2.connect(
        host="82.202.129.245",
        port=5432,
        sslmode="disable",
        dbname="neuro_base",
        user="elcom_user",
        password="Elcom_1998",
        target_session_attrs="read-write"
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Успешное подключение к PostgreSQL. Время подключения: {duration:.2f} сек")
        logger.debug(f"Подключение установлено: {connection}")
        
        return connection
    
    except Exception as e:
        print(f"Ошибка подключения к PostgreSQL: {e}")
        logger.error(f"Ошибка подключения к PostgreSQL: {str(e)}")
        logger.error(f"Время до ошибки: {duration:.2f} сек")
        return None
