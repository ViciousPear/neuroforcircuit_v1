from . import connection

def search_automatics(qf_object):
    connect_for_cursor = connection.connect_to_postgres()
    cursor_for_selection = connect_for_cursor.cursor()
    query = f'SELECT * FROM select_names_prices({repr(qf_object.Current.strip())}, {repr(qf_object.Voltage.strip())}, {repr(qf_object.Current_Close.strip())}) ORDER BY CASE WHEN full_name_a LIKE \'%ESQ%\' THEN 1 ELSE 0 END DESC'
    try:
        cursor_for_selection.execute(query)
        existing_record = cursor_for_selection.fetchall()
        connect_for_cursor.commit()
        return existing_record
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return
    
def search_wh():
    connect_for_cursor = connection.connect_to_postgres()
    cursor_for_selection = connect_for_cursor.cursor()
    query = f'SELECT * FROM WH_Counter LIMIT 10'
    try:
        cursor_for_selection.execute(query)
        existing_record = cursor_for_selection.fetchall()
        connect_for_cursor.commit()
        return existing_record
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return
    

def search_trans():
    connect_for_cursor = connection.connect_to_postgres()
    cursor_for_selection = connect_for_cursor.cursor()
    query = f'SELECT * FROM Transformator LIMIT 10'
    try:
        cursor_for_selection.execute(query)
        existing_record = cursor_for_selection.fetchall()
        connect_for_cursor.commit()
        return existing_record
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return