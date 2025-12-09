from . import connection
from typing import List

def search_automatics(qf_object: dict) -> List[dict]:
    connect_for_cursor = connection.connect_to_postgres()
    cursor_for_selection = connect_for_cursor.cursor()
    query = """
    SELECT * FROM search_automatics(
        %s, %s, %s, %s, %s, %s, 'ESQ'
    ) 
    """
    #ORDER BY CASE WHEN full_name_a LIKE '%%ESQ%%' THEN 1 END
    params = (
        qf_object["Current"].strip(),
        qf_object["Current_Close"].strip(),
        qf_object["Mounting_Type"].strip(),
        qf_object["Polus"].strip(),
        qf_object["Name"].strip(),
        qf_object["Voltage"].strip()
    )
    try:
        cursor_for_selection.execute(query, params)
        existing_record = cursor_for_selection.fetchall()
        connect_for_cursor.commit()
        results = convert_breakers(existing_record)
        return results
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return
    
def search_wh():
    connect_for_cursor = connection.connect_to_postgres()
    cursor_for_selection = connect_for_cursor.cursor()
    query = f'SELECT * FROM WH_Counter'
    try:
        cursor_for_selection.execute(query)
        existing_record = cursor_for_selection.fetchall()
        connect_for_cursor.commit()
        results = convert_breakers(existing_record)
        return results
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return
    

def search_trans():
    connect_for_cursor = connection.connect_to_postgres()
    cursor_for_selection = connect_for_cursor.cursor()
    query = f'SELECT * FROM Transformator'
    try:
        cursor_for_selection.execute(query)
        existing_record = cursor_for_selection.fetchall()
        connect_for_cursor.commit()
        results = convert_breakers(existing_record)
        return results
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return

def convert_breakers(raw_data: list) -> list[dict]:
    """Преобразует список кортежей в список словарей"""
    result = []
    for item in raw_data:
        if len(item) >= 3:  # Проверяем, что кортеж содержит все необходимые элементы
            breaker = {
                'id': str(item[0]),      # Первый элемент - id
                'name': str(item[1]),    # Второй элемент - название
                'price': str(item[2])    # Третий элемент - цена (в виде строки)
            }
            result.append(breaker)
    return result

