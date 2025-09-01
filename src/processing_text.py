import re
import qf, ta


def fix_current_value(text):
    # Удаляем "А" в конце, если есть (русская/латинская)
    if text == '':
        return text
    text = re.sub(r'[AА]$', '', text.strip())
    # Оставляем только цифры и дефис
    text = re.sub(r'[^0-9-]', '', text)
    # Разделяем по дефису
    parts = text.split('-')
    try:
        if len(parts) != 2:
            return text + 'А'  # Если дефиса нет, возвращаем как есть + "А"
            
        if(int(parts[0]) > int(parts[1])):
            parts[0] = parts[0][:-1]
            
        return f'{parts[0]}-{parts[1]}А'
    except:
        return text+'А'
       
    

def fix_current(text):
    text = re.sub(r'[^0-9AА]', '', text)
    text.rstrip()
    if text.isdigit() and len(text) >= 4:
        text = text[:-1] + 'А'
    else:
        text = text + 'А'
    return text

def expand_voltage(value):
    rules = [
        (r'AC\d*/\d*[BВ]$', 'AC380/415', 'В'),
        (r'400AC', '400 ', 'AC')
    ]

    for pattern, multiplier, unit in rules:
        if re.fullmatch(pattern, value):
            return f"{multiplier}{unit}"
    
    return value


def expand_current_value(short_value):
    """
    Преобразует сокращённое обозначение тока в полное значение.
    
    Параметры:
        short_value (str): Сокращённое значение (например, "000А", "00А", "0кА").
    
    Возвращает:
        str: Полное значение (например, "4000А", "100А", "50кА").
    """
    if len(short_value) == 0:  # или if not short_value:
        return short_value
    # Удаляем пробелы и приводим к верхнему регистру для унификации
    if (short_value[0] == '0' and len(short_value)>2):
        short_value = short_value[1:]
    
    # Ключ: (шаблон_сокращения, множитель, единица_измерения)
    rules = [
        (r'^0{3}[AА]$', 4000, 'А'),  # 000А → 4000А
        (r'^0{2}[AА]$', 100, 'А'),    # 00А → 100А
        (r'^0кА$', 50, 'кА'),      # 0кА → 50кА
        (r'^\w*0{3}кA$', 4000, 'кА'), # 000KA → 4000KA (латиница)
        (r'^0{2}кA$', 100, 'кA'),  # 00KA → 100KA
        (r'^0кA$', 50, 'кА'),       # 0KA → 50KA
        (r'^5кА', 35, 'кА'),
        (r'^8кА', 85, 'кА'),
        (r'^6кА', 65, 'кА'),
        (r'^2кА', 25, 'кА'),
        (r'^0-00[AА]$', '160-400', 'А'),
        (r'^[3-9]\d{2}-[0-3]\d{2}A$', '100-250', 'А'),
        (r'^63[0-9][AА]$', 630, 'А'),
        (r'^[0-6][AА]$', 6.3, 'А'),
        (r'^кА$', 25, 'кА'),
        (r'^\d*[AА]\d[AА]', 6.3, 'А'),
        (r'\b[0]{1,3}-\d{1,4}[AА]\d*\w*', '100-250', 'А'),
        (r'^\d-\d[AА]$', '40-100', 'А'),
        (r'^44-100[AА]$', '40-100', 'А'),
        (r'^00кА$', 100, 'кА'),
        (r'^[^0-9-]*(-?\d{4,}[AА])$', 5000, 'А'),
        (r'^[AА][0-9]', 10, 'А'),
        (r'^[AА]\d{2,3}', 50, 'А'),
        (r'^[^1]\d{2}кА', short_value[1:-2], 'кА'),
        (r'^\d*[4]$', short_value[:-1], 'А'),
        (r'^0-125[AА]', '50-125', 'А'),
        (r'^\w{1}\d*-\d*[AА]$',short_value[0:-1], 'А'),
        (r'^[AА]{1,3}', 6.3, 'А'),
        (r'^\d*[AА]{2}',short_value[:-2], 'А'),
        (r'^\s*00[АаA]\s*$', 1000, 'А')
    ]
    
    for pattern, multiplier, unit in rules:
        if re.fullmatch(pattern, short_value):
            return f"{multiplier}{unit}"
        

        
    complex_pattern = r'(?:^|[^AА0-9])(?P<value>\d{2,4})(?P<unit>[AА])(?:\d{2,4}(?P<unit2>[AА]))?|(?P<range>\d+-\d+[AА])'
    
    match = re.search(complex_pattern, short_value, re.IGNORECASE)
    if match:
        # Обработка слипшихся значений (20A210A)
        if match.group('unit2'):
            return f"{match.group('value')}{match.group('unit')}"
        
        # Обработка диапазонов (100-200A)
        if match.group('range'):
            return match.group('range')
        
        # Обработка одиночных значений (210A)
        return f"{match.group('value')}{match.group('unit')}"
    
    # Если шаблон не совпал, возвращаем исходное значение
    return short_value

def _correct_single_transformer(text):
    if not text:
        return text
    
    text = text.strip().upper()
    
    rules = [
        # ТТАТ → 1ТА1 (первая Т → 1, вторая Т → 1)
        (r'^ТТАТ$', '1ТА1'),
        (r'^ТТА([1-6])$', r'1ТА\1'),
        (r'^Т([1-4])АТ$', r'\1ТА1'),
        
        # Правила для 31А1 → 3ТА1 (перестановка цифр)
        (r'^31[АA]([1-6])$', r'3ТА\1'),  # Добавлена латинская A
        (r'^13[АA]([1-6])$', r'1ТА\1'),  # Добавлена латинская A
        
        # Правила для 21А3 → 2ТА3 (лишняя цифра 1) - ИСПРАВЛЕНО!
        (r'^([1-4])1[АA]([1-6])$', r'\1ТА\2'),  # Добавлена латинская A
        (r'^([1-4])2[АA]([1-6])$', r'\1ТА\2'),  # Добавлена латинская A
        (r'^([1-4])3[АA]([1-6])$', r'\1ТА\2'),  # Добавлена латинская A
        (r'^([1-4])\d[АA]([1-6])$', r'\1ТА\2'),  # Общее правило для любой цифры + латинская A
        
        # Правила для 1ТАТ → 1ТА1 (последняя Т → 1)
        (r'^([1-4])ТАТ$', r'\1ТА1'),
        (r'^([1-4])[ТT]A[TТ]$', r'\1ТА1'),  # Добавлено для латинской A
        
        # Правила для 11А3 → 1ТА3
        (r'^11[АA]([1-6])$', r'1ТА\1'),  # Добавлена латинская A
        
        # Общие правила замены
        (r'^ТТ$', '1ТА1'),
        (r'^ТА$', '1ТА1'),
        (r'^[ТT]([1-4])$', r'\1ТА1'),  # Добавлена латинская T
        (r'^([1-4])[ТT]$', r'\1ТА1'),  # Добавлена латинская T
        
        # Замена А на ТА (исправлено для латинских букв)
        (r'^([1-4])[АA]([1-6])$', r'\1ТА\2'),  # Добавлена латинская A
        
        # Замена лишних символов (исправлено для латинских букв)
        (r'^([1-4])[ТT][ТT]{1,2}([1-6])$', r'\1ТА\2'),  # Добавлена латинская T
        
        # Корректные значения оставляем как есть (исправлено для латинских букв)
        (r'^[1-4][ТT][АA][1-6]$', lambda x: x),
        
        # Правила для смешанных символов (исправлено для латинских букв)
        (r'^([1-4])[ТT][AА][TТ]$', r'\1ТА1'),
        
        # Новые правила для различных ошибок (исправлено для латинских букв)
        (r'^([1-4])[ТT][АA](\d)$', r'\1ТА\2'),  # 2ТА1 → 2ТА1
        (r'^([1-4])[ТT](\d)$', r'\1ТА\2'),      # 2Т1 → 2ТА1
        (r'^([1-4])[АA](\d)$', r'\1ТА\2'),      # 2А1 → 2ТА1
        
        # Дополнительное правило для случаев типа 21A3
        (r'^([1-4])(\d)[АA](\d)$', r'\1ТА\3'),  # 21A3 → 2ТА3
    ]

    for pattern, replacement in rules:
        match = re.fullmatch(pattern, text)
        if match:
            if callable(replacement):
                return replacement(text)
            return re.sub(pattern, replacement, text)
    
    # Если не найдено подходящее правило, пытаемся извлечь базовую структуру
    base_match = re.match(r'^.*?([1-4]).*?([1-6]).*$', text)
    if base_match:
        return f"{base_match.group(1)}ТА{base_match.group(2)}"
    
    return text

def ta_current_value(text):
    if not text or len(text.strip()) == 0:
        return
    
    text = text.strip().upper()
    range_pattern = r'^(.+?)\s*[-–—]\s*(.+)$'
    range_match = re.match(range_pattern, text)

    if range_match:
        # Обрабатываем обе части диапазона отдельно
        part1 = range_match.group(1)
        part2 = range_match.group(2)
        
        corrected_part1 = _correct_single_transformer(part1)
        corrected_part2 = _correct_single_transformer(part2)
        
        return f"{corrected_part1}-{corrected_part2}"


    return _correct_single_transformer(text)


def search_qf(text):
    id_qf = r"\b\d*[OQ0]F[TG]?\d*\.?\d*\b"  # Идентификатор (например, QF19)
    current_pattern = r"\b\w{1}?\d+\s*-\s*\d*[oO0]\d*\s*[AА]\b|\b\d+\s*[AА4]\b"
    voltage_pattern = r"([AА][CС]\d+/\d{3}|[AА][CС]\s*\d{3}|\b\d{3}[AА][CС])"
    current_close_pattern = r"\d{2,4}(?=\D*kA)"
    device_id = ''.join(re.findall(id_qf, text))
    current_range = ''.join(re.findall(current_pattern, text)) 
    current_voltage = ''.join(re.findall(voltage_pattern, text))
    if re.findall(voltage_pattern, text) and re.fullmatch(r'[AА][CС]\d+/\d{3}|[AА][CС]\s*\d{3}', current_voltage):
      current_voltage = current_voltage + 'В' 

    current_close = ''.join(re.findall(current_close_pattern, text))[1:] + 'кА' if re.findall(current_close_pattern, text) else ''
    if re.fullmatch(r'\b\d+\s*[AА]?\b', current_range):
        current_range = fix_current(current_range)
    else:
        current_range = fix_current_value(current_range)
    if (current_range != '' or current_voltage != '' or current_close != ''):
        current_range = expand_current_value(current_range)
        current_close = expand_current_value(current_close)
        current_voltage = expand_voltage(current_voltage)
        new_qf = qf.create_qf(device_id, current_range, current_voltage, current_close)

        return new_qf
    return

def search_ta(text):
    ta_text =  r"[\dТT]*[ТT]?[АA]+[\dТT]*-[\dТT]*[ТT]?[АA]+[\dТT]*"
    ta_exc = ''.join(re.findall(ta_text, text))
    if (ta_exc != ''):
        ta_exc = ta_current_value(ta_exc)
        new_ta = ta.create_ta(ta_exc)
        return new_ta
    return

if __name__ == "__main__":
    test_cases = [
    # Корректные диапазоны
    "1ТА1-1ТА3",
    "2ТА4-2ТА6", 
    "3ТА1-3ТА3",
    
    # Ошибочные случаи из примеров
    "31А1-31А3",
    "1ТАТ-11А3", 
    "ТТАТ-1ТА3",  # Должно стать: 1ТА1 - 1ТА3
    "TTA1-1TA3",
    "2TA1-21A3",
    
    # Одиночные значения
    "1ТА1", "2ТА4", "31А1", "1ТАТ", "ТТАТ", "11А3",
    
    # Дополнительные тесты
    "ТТА1", "Т1АТ", "1ТТ3", "ТА2", "Т3",
    "100А", "250-630А", "5кА"
    ]
    ta_list = []
    print("Тестирование функции:")
    for test in test_cases:
        corrected = ta_current_value(test)
        print(f"'{test}' -> '{corrected}'")
        ta_correct = search_ta(test)
        if ta_correct != None:
            ta_list.append(ta_correct)
    for ta_object in ta_list:
        print(f"Имя:{ta_object.ta_name}, Количество: {ta_object.ta_quantity}")

