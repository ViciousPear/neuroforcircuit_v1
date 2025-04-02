import psycopg2 

def connect_to_postgres():
    """
    Подключается к PostgreSQL (neuro_base).
    """
    try:
        #это надо перенести в отдельный файл с паролями и прочим
        connection = psycopg2.connect(
            dbname="neuro_base",
            user="postgres",
            password="Pe_Pe_856",
            host="localhost",  # или IP сервера
            port="5432"
        )
        return connection
    except Exception as e:
        print(f"Ошибка подключения к PostgreSQL: {e}")
        return None
