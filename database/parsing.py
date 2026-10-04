import requests
import xmltodict
from typing import List, Dict
from dotenv import load_dotenv
import os
import connection
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

dotenv_path = '/app/db_connect/.env'
load_dotenv(dotenv_path)


def get_token():
    """Подключение к каталогу"""
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
            return None
    except Exception as e: 
        print(f"Произошла ошибка: {e}")
        return None


# Конфигурация API
API_BASE_URL = 'https://api.elcomspb.ru/GetOffers'


# Категории для парсинга
CATEGORIES = {
    "vozdushnye": [212, 213, 236, 301, 300],       
    "litoy_korpus": [216, 217, 218],
    "modulniye": [221, 222, 223]  
}


def should_include_item(name: str, price: float) -> bool:
    """
    Проверяет, должен ли элемент быть включен в базу данных
    Возвращает False если элемент нужно отфильтровать
    """
    # Проверка на нулевую цену
    if price == 0:
        return False
    
    # Проверка на наличие обязательных слов в названии
    if "Воздушный" not in name and "Автоматический" not in name:
        return False
    
    # Если все проверки пройдены, включаем элемент
    return True


def get_api_data(category_ids: List[int]) -> List[Dict]:
    """Получает данные с API для указанных категорий с фильтрацией"""
    API_TOKEN = get_token()
    if not API_TOKEN:
        print("Не удалось получить токен авторизации")
        return None
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/xml"
    }
    
    all_results = []
    filtered_count = 0
    
    for category_id in category_ids:
        url = f"{API_BASE_URL}?pageSize=1000&offset=0&category={category_id}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            xml_data = xmltodict.parse(response.text)
            offers = xml_data["offers"]["offer"]
            
            # Убеждаемся, что offers - это список
            if not isinstance(offers, list):
                offers = [offers]
            
            for item in offers:
                name = item["name"]
                price = float(item["price"])
                
                # Фильтрация элементов перед добавлением
                if should_include_item(name, price):
                    all_results.append({
                        "article": item["number"],
                        "name": name,
                        "price": price,
                    })
                else:
                    filtered_count += 1
                    print(f"Отфильтровано: {item['number']} - {name} (price: {price})")
                
        except Exception as e:
            print(f"Ошибка при обработке категории {category_id}: {str(e)}")
            continue  
    
    print(f"\nСтатистика фильтрации для категории:")
    print(f"  Всего получено с API: {len(all_results) + filtered_count}")
    print(f"  Отфильтровано: {filtered_count}")
    print(f"  Добавлено в БД: {len(all_results)}")
    
    return all_results


def create_table(conn):
    """Создает таблицу если её нет"""
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS automatics (
            article_a VARCHAR(50) PRIMARY KEY,
            full_name_a TEXT NOT NULL,
            price NUMERIC(10, 2) NOT NULL,
            current_rating VARCHAR(50),
            current_close NUMERIC(10, 2),
            poles INTEGER,
            mounting_type INTEGER,
            short_name VARCHAR(100),
            production VARCHAR(50),
            voltage VARCHAR(50)
        );
        """)
        conn.commit()
        print("Таблица automatics проверена/создана")


def update_database(conn, data: List[Dict]):
    """Обновляет данные в БД"""
    added_count = 0
    updated_count = 0
    
    with conn.cursor() as cursor:
        for item in data:
            # Проверка на существование записи
            cursor.execute(
                "SELECT price FROM automatics WHERE article_a = %s",
                (item["article"],))
            existing = cursor.fetchone()
            
            if existing:
                if float(existing[0]) != item["price"]:
                    cursor.execute("""
                    UPDATE automatics SET 
                        price = %s,
                        full_name_a = %s
                    WHERE article_a = %s;
                    """, (item["price"], item["name"], item["article"]))
                    updated_count += 1
                    print(f"Обновлен: {item['article']} - {item['name']}")
                else:
                    print(f"Без изменений: {item['article']} - {item['name']}")
            else:
                try:
                    cursor.execute("""
                    INSERT INTO automatics (article_a, full_name_a, price) 
                    VALUES (%s, %s, %s);
                    """, (item["article"], item["name"], item["price"]))
                    added_count += 1
                    print(f"Добавлен: {item['article']} - {item['name']}")
                except Exception as e:
                    print(f"Ошибка при вставке {item['article']}: {e}")
                    conn.rollback()
                    continue
       
        conn.commit()
        
    print(f"\nСтатистика обновления БД:")
    print(f"  Добавлено новых: {added_count}")
    print(f"  Обновлено существующих: {updated_count}")
    
    return added_count + updated_count


def update_characteristics(conn):
    """Обновляет характеристики для записей, у которых они еще не заполнены"""
    print("\n--- Обновление характеристик ---")
    
    update_query = """
    WITH matches AS (
        SELECT 
            article_a,
            full_name_a,
            
            -- Извлекаем все регулярные выражения один раз в CTE
            (regexp_matches(full_name_a, '(TMD|TMF|E3)[\\s]*(\\d{1,4}(?:[.,]\\d)?-\\d{1,4}[AА]|\\d{1,4}(?:[.,]\\d)?[AА])', 'i')) as match1,
            (regexp_matches(full_name_a, '(TMD|TMF|E3)(\\d{1,4}(?:[.,]\\d)?-\\d{1,4}[AА]|\\d{1,4}(?:[.,]\\d)?[AА])', 'i')) as match2,
            (regexp_matches(full_name_a, '\\([^)]*?(\\d{1,4}(?:[.,]\\d)?-\\d{1,4}[AА]|\\d{1,4}(?:[.,]\\d)?[AА])[^)]*?(кА|kA|полюс|non)', 'i')) as match3,
            (regexp_matches(full_name_a, '[\\s\\(,](\\d{1,4}(?:[.,]\\d)?-\\d{1,4}[AА]|\\d{1,4}(?:[.,]\\d)?[AА])\\s', 'i')) as match4,
            (regexp_matches(full_name_a, '(?:^|[\\s\\(,])(\\d{2,4}(?:[.,]\\d)?-\\d{2,4}[AА]|\\d{2,4}(?:[.,]\\d)?[AА])(?:[\\s\\),]|кА|kA|$)', 'i')) as match5,
            (regexp_matches(full_name_a, '(\\d{1,4}[.,]\\d{1,2})(?=\\s*[kк][AА])', 'i')) as match6_decimal,
            (regexp_matches(full_name_a, '(\\d{1,4})(?=\\s*[kк][AА])', 'i')) as match6_int,
            (regexp_matches(full_name_a, '[BВ][АA]\\s*\\d{1,4}(?:-\\d{1,4})?(?:/\\d{1,4})?[A-ZА-Я]*', 'i')) as match_short1,
            (regexp_matches(full_name_a, '[HН]G[A-Z]?\\s*\\d{2,4}[A-Z]?(?:-[A-Z])?', 'i')) as match_short2,
            (regexp_matches(full_name_a, 'U[A-Z]{1,3}\\d{2,4}[A-Z]?', 'i')) as match_short3,
            (regexp_matches(full_name_a, '\\d{3,4}\\s?[AА][CС]|[AА][CС]\\d{3,4}/\\d{3,4}', 'i')) as match_voltage,
            
            -- Добавляем дополнительные regexp_matches для poles
            (regexp_matches(full_name_a, '\\s([1])[A-HА-В]\\s', 'i')) as pole_match1,
            (regexp_matches(full_name_a, '\\s([2])[A-HА-В]\\s', 'i')) as pole_match2,
            (regexp_matches(full_name_a, '\\s([3])[A-HА-В]\\s', 'i')) as pole_match3,
            (regexp_matches(full_name_a, '\\s([4])[A-HА-В]\\s', 'i')) as pole_match4
        FROM automatics
        WHERE short_name IS NULL OR current_rating IS NULL
    )
    UPDATE automatics a
    SET 
        current_rating = CASE
            -- 1. Сначала ищем после технических обозначений TMD, TMF, E3
            WHEN a.full_name_a ~ '(TMD|TMF|E3)[\\s]?\\d' THEN
                COALESCE(
                    m.match1[2],
                    m.match2[2]
                )
            
            -- 2. Ищем в скобках с характеристиками
            WHEN a.full_name_a ~ '\\(.*\\d{1,4}[AА].*(кА|kA|полюс|non)' THEN
                m.match3[1]
            
            -- 3. Ищем с пробелами вокруг
            WHEN a.full_name_a ~ '\\s\\d{1,4}[AА]\\s' AND a.full_name_a !~ '\\s[1234][AА]\\s[M2C]' THEN
                m.match4[1]
            
            -- 4. Общий поиск
            ELSE
                m.match5[1]
        END,
        
        current_close = COALESCE(
            CASE 
                WHEN m.match6_decimal[1] IS NOT NULL THEN
                    REPLACE(m.match6_decimal[1], ',', '.')::NUMERIC
                ELSE NULL
            END,
            m.match6_int[1]::NUMERIC,
            0
        ),
        
        poles = COALESCE(
            CASE 
                WHEN a.full_name_a ~ '3P|3 полюс|3 полюса|3 п\\.|3п\\.|3 пол\\.|3пол\\.|3 non\\.|3пол\\s|3пол[^ю]' THEN 3
                WHEN a.full_name_a ~ '4P|4 полюс|4 полюса|4 п\\.|4п\\.|4 пол\\.|4пол\\.' THEN 4
                WHEN a.full_name_a ~ '1P|1 полюс|1 п\\.|1п\\.|1 пол\\.|1пол\\.' THEN 1
                WHEN a.full_name_a ~ '2P|2 полюс|2 полюса|2 п\\.|2п\\.|2 пол\\.|2пол\\.' THEN 2
                ELSE NULL
            END,
            CASE 
                WHEN a.full_name_a ~ '\\s[1234][A-HА-В]\\s' THEN 
                    CASE 
                        WHEN m.pole_match1[1] = '1' THEN 1
                        WHEN m.pole_match2[1] = '2' THEN 2
                        WHEN m.pole_match3[1] = '3' THEN 3
                        WHEN m.pole_match4[1] = '4' THEN 4
                        ELSE NULL
                    END
                ELSE NULL
            END
        ),
        
        mounting_type = COALESCE(
            CASE 
                WHEN a.full_name_a ~ 'Втычной|втычной|Выкатной|выкатной|с корзиной' THEN 1
                WHEN a.full_name_a ~ 'Стационарный|стационарный|без корзины' THEN 0
                ELSE NULL
            END,
            CASE 
                WHEN a.full_name_a ~ 'Воздушный|воздушный' AND a.full_name_a ~ '\\s[1234]([BВD])\\s' THEN 1
                WHEN a.full_name_a !~ 'Воздушный|воздушный' AND a.full_name_a ~ '\\s[1234]([PРD])\\s' THEN 1
                ELSE NULL
            END,
            0
        ),
        
        short_name = regexp_replace(
            COALESCE(
                m.match_short1[1],
                m.match_short2[1],
                m.match_short3[1]
            ),
            '\\s+', '', 'g'
        ),
        
        production = CASE 
            WHEN a.full_name_a ~ 'ESQ' OR a.full_name_a ~ '[BВ][AА]\\s*\\d{1,4}\\s*-\\d{1,4}\\s*' THEN 'ESQ'
            ELSE 'Hyundai'
        END,
        
        voltage = m.match_voltage[1]

    FROM matches m
    WHERE a.article_a = m.article_a
    AND (a.short_name IS NULL OR a.current_rating IS NULL);
    """
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(update_query)
            updated_count = cursor.rowcount
            conn.commit()
            print(f"Обновлено характеристик: {updated_count} записей")
            return updated_count
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при обновлении характеристик: {e}")
        print(f"Текст ошибки: {str(e)}")
        return 0


def cleanup_old_records(conn):
    """Очищает старые записи, которые не соответствуют критериям (для обратной совместимости)"""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                DELETE FROM automatics 
                WHERE (full_name_a NOT LIKE '%Воздушный%' 
                AND full_name_a NOT LIKE '%Автоматический%')
                OR price = 0
                """)
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                print(f"Очистка старых записей: удалено {deleted_count} записей")
            return deleted_count
            
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при очистке старых записей: {e}")
        return 0


def verify_data_insertion(conn, expected_count):
    """Проверяет, сколько записей действительно добавилось"""
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM automatics")
            count = cursor.fetchone()[0]
            print(f"\n=== ПРОВЕРКА: В таблице {count} записей ===")
            return count
    except Exception as e:
        print(f"Ошибка при проверке данных: {e}")
        return 0


def single_parsing_iteration(conn):
    """Один цикл сбора и обработки данных"""
    print("\n" + "="*50)
    print(f"Начало цикла сбора данных с API в {datetime.now()}")
    print("="*50)
    
    all_data = []
    
    try:
        # Сбор данных для всех категорий
        for category_name, category_ids in CATEGORIES.items():
            print(f"\n--- Сбор данных для категории: {category_name} ---")
            category_data = get_api_data(category_ids)
            if category_data:  
                all_data.extend(category_data)
                print(f"Получено {len(category_data)} записей из категории {category_name}")
        
        print(f"\nВсего собрано данных: {len(all_data)} записей")
        
        if all_data: 
            # 1. Обновление основных данных в БД
            processed_count = update_database(conn, all_data)
            
            # 2. Обновление характеристик
            if processed_count > 0:
                updated_chars_count = update_characteristics(conn)
                print(f"\nИтого обработано: {processed_count} записей данных, {updated_chars_count} записей с обновленными характеристиками")
            
            # 3. Проверка
            verify_data_insertion(conn, len(all_data))
            
        else:
            print("Нет данных для обработки")
            
    except Exception as e:
        print(f"Ошибка в основном цикле: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print(f"Цикл сбора данных завершен в {datetime.now()}")
    print("="*50)


def run_parsing(conn):
    """Функция для APScheduler"""
    print(f"\n=== Запуск парсинга в {datetime.now()} ===")
    single_parsing_iteration(conn)


if __name__ == "__main__":
    conn = None
    try:
        conn = connection.connect_to_postgres()
        if not conn:
            raise RuntimeError("Не удалось подключиться к БД")
        
        # Создание таблицы при первой записи
        create_table(conn)
        
        # Первый запуск для проверки
        run_parsing(conn)
        
        # Настройка планировщика
        scheduler = BlockingScheduler()
        
        # Парсинг каждое воскресенье в 03:00
        scheduler.add_job(
            lambda: run_parsing(conn),  
            'cron',
            day_of_week='sun',
            hour=3,
            misfire_grace_time=3600
        )
        
        print("\nПланировщик запущен. Ожидание воскресенья 03:00...")
        scheduler.start()
        
    except (KeyboardInterrupt, SystemExit):
        print("\nОстановка планировщика...")
        if 'scheduler' in locals():
            scheduler.shutdown()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("Соединение с БД закрыто")