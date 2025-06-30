import psycopg2 
from dotenv import load_dotenv
import logging
from datetime import datetime
import os

load_dotenv()

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('postgres_connections.log'),  # Логи в файл
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
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        sslmode=os.getenv("POSTGRES_SSLMODE", "disable"),
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
