import requests
import xmltodict
from typing import List, Dict
from dotenv import load_dotenv
import os
from connection import connect_to_postgres
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

dotenv_path = '/app/db_connect/.env'
load_dotenv(dotenv_path)


def get_token():
    body = {
    'email': os.environ.get('login'),
    'password': os.environ.get('password')
    }
    auth_url = 'https://api.elcomspb.ru/Auth'

    try:
        response = requests.post(auth_url, json=body)
        if response.status_code == 200:
            token = response.json().get('auth_token')
            return token
        else:
            print(f"Произошла ошибка запроса: {response.status_code}")
            print(f"Ответ сервера: {response.text}")
            return
    except Exception as e: 
        print(f"Произошла ошибка: {e}")
        return
    
# Конфигурация API
API_BASE_URL = 'https://api.elcomspb.ru/GetOffers'


# Категории для парсинга
CATEGORIES = {
    "vozdushnye": [212, 236, 301, 300],       
    "litoy_korpus": [216, 217, 218] 
}

def get_api_data(category_ids: List[int]) -> List[Dict]:
    """Получает данные с API для указанных категорий"""

    API_TOKEN = get_token()
    if not API_TOKEN:
        print("Не удалось получить токен авторизации")
        return None
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/xml"
    }
    
    all_results = []
    
    for category_id in category_ids:
        url = f"{API_BASE_URL}?pageSize=1000&offset=0&category={category_id}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            xml_data = xmltodict.parse(response.text)
            offers = xml_data["offers"]["offer"]
            
            for item in offers:
                all_results.append({
                    "article": item["number"],
                    "name": item["name"],
                    "price": float(item["price"]),
                })
                
        except Exception as e:
            print(f"Ошибка при обработке категории {category_id}: {str(e)}")
            return None
    
    return all_results

def create_table(conn):
    """Создает таблицу если её нет"""
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS automatics (
            article VARCHAR(50) PRIMARY KEY,
            name TEXT NOT NULL,
            price NUMERIC(10, 2) NOT NULL,
            category_id INTEGER NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()
        print("Таблица automatics проверена/создана")

def update_database(conn, data: List[Dict]):
    """Обновляет данные в БД"""
    with conn.cursor() as cursor:
        for item in data:
            # Проверяем существование записи
            cursor.execute(
                "SELECT price FROM automatics WHERE automatics.article_a = %s",
                (item["article"],))
            existing = cursor.fetchone()
            
            if existing:
                # Обновляем если цена изменилась
                if float(existing[0]) != item["price"]:
                    cursor.execute("""
                    UPDATE Automatics SET price = %s,
                    full_name_a = %s 
                    WHERE article_a = %s;
                    """, (item["price"], item["name"], item["article"]))
                    print(f"Обновлен: {item['article']} - {item['name']}")
            else:
                # Вставляем новую запись
                cursor.execute("""
                INSERT INTO Automatics (article_a, full_name_a, price) 
                VALUES (%s, %s, %s) 
                ON CONFLICT DO NOTHING;
                """, (item["article"], item["name"], item["price"]))
                print(f"Добавлен: {item['article']} - {item['name']}")
       
        conn.commit()

def delete_from_database(conn):
    try:
        with conn.cursor() as cursor:
            # Сначала проверяем, что будем удалять
            cursor.execute("""
                SELECT full_name_a FROM Automatics 
                WHERE full_name_a NOT LIKE %s
                AND full_name_a NOT LIKE %s
                """, 
                ("%Воздушный%", "%Автоматический%"))
            rows_to_delete = cursor.fetchall()
            
            if not rows_to_delete:
                print("Нет записей для удаления")
                return
            
            print("Найдены записи для удаления:")
            for row in rows_to_delete:
                print(f"ID: {row[0]}, Название: {row[1]}")
            
            # Удаляем
            cursor.execute("""
                DELETE FROM Automatics 
                WHERE full_name_a NOT LIKE %s
                AND full_name_a NOT LIKE %s
                """, 
                ("%Воздушный%", "%Автоматический%"))
            
            conn.commit()  # Важно! Без этого изменения не сохранятся
            print(f"Удалено записей: {cursor.rowcount}")
            
    except Exception as e:
        conn.rollback()  # Откат в случае ошибки
        print(f"Ошибка: {e}")


def single_parsing_iteration(conn):
    """Один цикл сбора и обработки данных"""
    print("\nНачало цикла сбора данных с API...")
    try:
        # Собираем данные для всех категорий
        all_data = []
        for category_name, category_ids in CATEGORIES.items():
            print(f"Сбор данных для категории: {category_name}")
            category_data = get_api_data(category_ids)
            if category_data:  # Проверяем, что данные получены
                all_data.extend(category_data)
                print(f"Получено {len(category_data)} записей")
        
        if all_data:  # Если есть данные для обработки
            # Обновляем БД
            update_database(conn, all_data)
            print(f"Всего обработано {len(all_data)} записей")
            delete_from_database(conn)
        else:
            print("Нет данных для обработки")
            
    except Exception as e:
        print(f"Ошибка в основном цикле: {str(e)}")

def run_parsing(conn):
    """Функция для APScheduler"""
    print(f"\n=== Запуск парсинга в {datetime.now()} ===")
    single_parsing_iteration(conn)

if __name__ == "__main__":
    conn = None
    try:
        conn = connect_to_postgres()
        if not conn:
            raise RuntimeError("Не удалось подключиться к БД")
        
        # Создаем таблицу при первом запуске
        create_table(conn)
        
        # Настраиваем планировщик
        scheduler = BlockingScheduler()
        
        # Парсинг каждое воскресенье в 03:00
        scheduler.add_job(
            lambda: run_parsing(conn),  
            'cron',
            day_of_week='sun',
            hour=3,
            misfire_grace_time=3600
        )
        
        print("Планировщик запущен. Ожидание воскресенья 03:00...")
        scheduler.start()
        
    except (KeyboardInterrupt, SystemExit):
        print("\nОстановка планировщика...")
        if 'scheduler' in locals():
            scheduler.shutdown()
    finally:
        if conn:
            conn.close()
            print("Соединение с БД закрыто")