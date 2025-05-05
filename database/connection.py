import psycopg2 
from dotenv import load_dotenv
import os

dotenv_path = 'database/db_connect/.env.utf8'
load_dotenv(dotenv_path)
def connect_to_postgres():
    """
    Подключается к PostgreSQL (neuro_base).
    """
    try:
        
        connection = psycopg2.connect(
            dbname=os.environ.get("dbname"),
            user=os.environ.get("user"),
            password=os.environ.get("password"),
            host=os.environ.get("host"),  # или IP сервера
            port=os.environ.get("port")
        )
        return connection
    except Exception as e:
        print(f"Ошибка подключения к PostgreSQL: {e}")
        return None
