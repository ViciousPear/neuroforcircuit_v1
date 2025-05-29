import requests
import xmltodict
import time
from typing import List, Dict
from dotenv import load_dotenv
import os
from . import connection  

load_dotenv()

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
    "vozdushnye": [212, 213, 300],       
    "litoy_korpus": [216, 217, 218, 236, 301] 
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

def cyclic_api_parsing(interval=3600):
    """Циклический сбор данных с API"""
    conn = connection.connect_to_postgres()
    if not conn:
        return
   
    
    while True:
        print("\nНачало цикла сбора данных с API...")
        try:
            # Собираем данные для всех категорий
            all_data = []
            for category_name, category_ids in CATEGORIES.items():
                print(f"Сбор данных для категории: {category_name}")
                category_data = get_api_data(category_ids)
                all_data.extend(category_data)
                print(f"Получено {len(category_data)} записей")
            
            # Обновляем БД
            update_database(conn, all_data)
            print(f"Всего обработано {len(all_data)} записей")
            
        except Exception as e:
            print(f"Ошибка в основном цикле: {str(e)}")
        
        print(f"Ожидание следующего цикла ({interval} сек)...")
        time.sleep(interval)

if __name__ == "__main__":
    cyclic_api_parsing(interval=3600)  # Запуск с интервалом 1 час
