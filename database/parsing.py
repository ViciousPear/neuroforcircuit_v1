import requests
from bs4 import BeautifulSoup
import re
import time
import psycopg2 
from . import connection

# Список сайтов для парсинга
sites = [
    {
        "name": "Воздушные автоматические выключатели",
        "base_url": "https://www.elcomspb.ru/retail/nizkovoltnoe-0-4kv-i-vysokovoltnoe-6-35kv-oborudov/vozdushnye-avtomaticheskie-vyklyuchateli/?PAGEN_1=",
        "pages": 15
    },
    {
        "name": "Автоматы в литом корпусе",
        "base_url": "https://www.elcomspb.ru/retail/nizkovoltnoe-0-4kv-i-vysokovoltnoe-6-35kv-oborudov/avtomaticheskie-vyklyuchateli-v-litom-korpuse/?PAGEN_1=",
        "pages": 16
    }
]

def parse_sites():
    """
    Функция для парсинга всех сайтов из списка.
    Возвращает список кортежей: [(артикул, название, цена), ...]
    """
    data = []

    for site in sites:
        print(f"Начинаем парсинг {site['name']}...")
        for n in range(1, site["pages"] + 1):
            url = f"{site['base_url']}{n}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                product_containers = soup.find_all('div', class_='product-table__product')

                if product_containers:
                    print(f"Обработка страницы {n} ({site['name']})...")
                    for container in product_containers:
                        # Парсим артикул
                        article_element = container.find('a', class_='product-table__article')
                        article = article_element.text.strip() if article_element else None

                        # Парсим название
                        name_element = container.find('a', class_='product-table__title')
                        name = name_element.text.strip() if name_element else None

                        # Парсим цену
                        price_element = container.find('strong', class_='thisPrice')
                        if price_element:
                            raw_price = price_element.text.strip()
                            clean_price = re.sub(r'[^\d.,]', '', raw_price).replace(',', '.')
                            price = float(clean_price) if clean_price else None
                        else:
                            price = None

                        if article and name and price is not None:
                            data.append((article, name, price))
                else:
                    print(f"Продукты не найдены на странице {n} ({site['name']}).")
            else:
                print(f"Ошибка при загрузке страницы {n} ({site['name']}): {response.status_code}")

    return data




def create_table(connection):
    """
    Создает таблицу Automatics, если её нет.
    """
    connect_for_cursor = connection.connect_to_postgres()
    cursor = connect_for_cursor.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Automatics (
        article VARCHAR(40) PRIMARY KEY,
        full_name TEXT,
        price NUMERIC(10, 2)
    );
    """
    try:
        cursor.execute(create_table_query)
        connection.commit()
        print("Таблица Automatics создана или уже существует.")
    except Exception as e:
        print(f"Ошибка при создании таблицы: {e}")


def update_or_insert_data(connection, data):
    """
    Обновляет или добавляет данные в таблицу Automatics.
    """
    cursor = connection.cursor()

    for article, name, price in data:
        # Проверяем, есть ли уже такой артикул
        cursor.execute(
            "SELECT full_name_a price FROM Automatics WHERE article_a = %s;",
            (article,)
        )
        existing_record = cursor.fetchone()

        if existing_record:
            # Если цена изменилась — обновляем
            if existing_record[0] != price:
                cursor.execute(
                    "UPDATE Automatics SET price = %s, full_name_a = %s WHERE article_a = %s;",
                    (price, name, article)
                )
                print(f"Обновлена запись: {article} ({name}) — {price} руб.")
        else:
            # Если артикула нет — добавляем
            cursor.execute(
                "INSERT INTO Automatics (article_a, full_name_a, price) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                (article, name, price)
            )
            print(f"Добавлена новая запись: {article} ({name}) — {price} руб.")

    connection.commit()


def cyclic_parsing(interval=3600):
    """
    Циклический парсинг с заданным интервалом (по умолчанию — 1 час).
    """
    connection = connection.connect_to_postgres()
    if not connection:
        return


    while True:
        print(f"\nЗапуск парсинга")
        try:
            parsed_data = parse_sites()
            update_or_insert_data(connection, parsed_data)
            print(f"Парсинг завершен. Следующий запуск через {interval} секунд.")
        except Exception as e:
            print(f"Ошибка при парсинге: {e}")
        
        time.sleep(interval)


if __name__ == "__main__":
    cyclic_parsing(interval=3600)  # Проверка каждые 60 минут