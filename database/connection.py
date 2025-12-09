import psycopg2 
from dotenv import load_dotenv
import logging
from datetime import datetime
import os

dotenv_path = '/app/database/.env'
load_dotenv(dotenv_path)
# load_dotenv()

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
    """Подключается к PostgreSQL (neuro_base)."""
    start_time = datetime.now()
    
    try:
        logger.info("Попытка подключения к PostgreSQL...")
        logger.debug(f"Параметры подключения: host={os.getenv('POSTGRES_HOST')}, ...")

        connection = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            sslmode=os.getenv("POSTGRES_SSLMODE")
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Успешное подключение. Время: {duration:.2f} сек")
        return connection

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Ошибка подключения: {e}")
        logger.info(f"Время попытки: {duration:.2f} сек")
        return None

connect_to_postgres()