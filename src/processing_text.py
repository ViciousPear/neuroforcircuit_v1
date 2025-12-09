import re
import qf, ta
# from . import qf
# from . import ta


def fix_merged_currents(text):
    """
    Обрабатывает слипшиеся значения токов, например:
    630630А → 630А
    250160А → 160-250А
    630500А → 500-630А
    250A180A → 180-250А
    400A280A → 280-400А
    """
    if not text:
        return text
    
    
    # Удаляем ВСЕ буквы "A" и "А" и пробелы
    text = text.strip().upper()
    text_without_all_a = re.sub(r'[AА]', '', text)
    
    
    # Проверяем, содержит ли текст тире
    if '-' in text or not re.match(r'^\d+$', text_without_all_a):
        return text
    
    # Проверяем длину
    if len(text_without_all_a) < 6:
        return text + 'А' if not text.endswith('А') and not text.endswith('A') else text
    
    
    # Делим текст пополам
    mid = len(text_without_all_a) // 2
    first_half = text_without_all_a[:mid]
    second_half = text_without_all_a[mid:]
    
    
    # Для нечётной длины добавляем балансировку
    if len(text_without_all_a) % 2 != 0:
        print(f"DEBUG: Нечётная длина, пробуем варианты")
        # Пробуем разные варианты разделения
        variants = [
            (text_without_all_a[:mid], text_without_all_a[mid:]),      # стандартное
            (text_without_all_a[:mid+1], text_without_all_a[mid+1:]),  # смещённое
        ]
        
        for first, second in variants:
            if first and second and first.isdigit() and second.isdigit():
                first_half, second_half = first, second
                break
    
    # Если части не являются числами, возвращаем исходный текст
    if not first_half.isdigit() or not second_half.isdigit():
        print(f"DEBUG: Части не цифровые")
        return text
    
    # Преобразуем в числа
    try:
        num1 = int(first_half)
        num2 = int(second_half)
    except ValueError:
        return text
    
    # Если числа равны
    if num1 == num2:
        result = f"{num1}А"
        return result
    
    # Определяем меньшее и большее
    smaller = min(num1, num2)
    larger = max(num1, num2)
    
    # Возвращаем в формате "меньшее-большееА"
    result = f"{smaller}-{larger}А"
    return result

def fix_current_value(text):
    # Удаляем "А" в конце, если есть (русская/латинская)
    if text == '':
        return text
    
    original_text = text.strip()
    # Удаляем ВСЕ буквы A/А для проверки
    text_without_all_a = re.sub(r'[AА]', '', original_text)
    
    
    # Сначала проверяем на слипшиеся значения (длинные цифровые последовательности)
    if len(text_without_all_a) >= 6 and re.match(r'^\d+$', text_without_all_a):
        fixed_merged = fix_merged_currents(original_text)
        if fixed_merged != original_text:  # Если функция что-то изменила
            return fixed_merged
    
    # Если длина <= 5 и нет тире - возвращаем как есть + "А"
    if len(text_without_all_a) <= 5 and '-' not in original_text:
        # Оставляем только цифры и дефис
        cleaned = re.sub(r'[^0-9-]', '', original_text)
        if cleaned and not (cleaned.endswith('А') or cleaned.endswith('A')):
            cleaned = cleaned + 'А'
        return cleaned
    
    # Оставляем только цифры и дефис
    text_cleaned = re.sub(r'[^0-9-]', '', original_text)
    
    # Разделяем по дефису
    parts = text_cleaned.split('-')
    
    try:
        if len(parts) != 2:
            # Если нет тире, добавляем А в конце если нет
            if text_cleaned and not (text_cleaned.endswith('А') or text_cleaned.endswith('A')):
                text_cleaned = text_cleaned + 'А'
            return text_cleaned
            
        if(int(parts[0]) > int(parts[1])):
            parts[0] = parts[0][:-1]
            
        return f'{parts[0]}-{parts[1]}А'
    except:
        if text_cleaned and not (text_cleaned.endswith('А') or text_cleaned.endswith('A')):
            text_cleaned = text_cleaned + 'А'
        return text_cleaned
       
    

def fix_current(text):
    # Удаляем всё, кроме цифр и букв A/А
    text = re.sub(r'[^0-9AА]', '', text)
    # Удаляем пробелы
    text = text.strip()
    
    # Если строка пустая
    if not text:
        return 
    
    # Если уже заканчивается на А, оставляем как есть
    if text.endswith('А') or text.endswith('A'):
        return text
    
    # Добавляем А в конце
    return text + 'А'

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
    # if (short_value[0] == '0' and len(short_value)>2):
    #     short_value = short_value[1:]
    
    # Ключ: (шаблон_сокращения, множитель, единица_измерения)
    rules = [
        (r'^0{3}[AА]$', 4000, 'А'),  # 000А → 4000А
        (r'^0{2}[AА]$', 100, 'А'),     # 00А → 100А
        (r'^[0-9]{1,3}0[1-9][AА]$', short_value[0:2], 'А'),     # 1003А -> 10А
        (r'[BВCС]?\d{1,4}',short_value[0:], 'А'),
        (r'[BВCС]?\d{1,4}[AА]?',short_value[0:-1], 'А'),
        (r'^\d0{2}[QO]0[AА]$', short_value[0]+"000", 'А'),  # [число]0Q0А → [число]000А
        (r'^\d0[QO][AА]$', short_value[0]+"0", 'А'),  # [число]QА → [число]0А
        (r'^\d0[QO][AА]$', short_value[0]+"0", 'А'),  # [число]QА → [число]0А
        (r'^0-100[AА]$', '40-100', 'А'),
        (r'^\d{2}[4][AА]$', short_value[:-2], 'А'),
        (r'^04[AА]$', 20, 'А'),
        (r'^0кА$', 50, 'кА'),      # 0кА → 50кА
        (r'^0{2}кA$', 100, 'кA'),  # 00KA → 100KA
        (r'^0кA$', 50, 'кА'),       # 0KA → 50KA
        (r'^5кА', 35, 'кА'),
        (r'^8кА', 85, 'кА'),
        (r'^6кА', 65, 'кА'),
        (r'^2кА', 25, 'кА'),
        (r'^[1-9][0-9]{2}кА', 50, 'кА'),
        (r'^0-00[AА]$', '160-400', 'А'),
        (r'[2-4][2-9]0-400[AА]$', '320-400', 'А'),
        (r'[4-6][0-9]0-630[AА]$', '504-630', 'А'),
        (r'[1-9][0-9]{3}-630[AА]$', '252-630', 'А'),
        (r'1[1-9][0-9]-250[AА]$', '200-250', 'А'),
        (r'[4-9][0-9]{2}-800[AА]$', '800', 'А'),
        (r'^[3-9]\d{2}-[0-3]\d{2}A$', '100-250', 'А'),
        (r'^63[0-9][AА]$', 630, 'А'),
        (r'^[0-6][AА]$', 6, 'А'),
        (r'^кА$', 25, 'кА'),
        (r'^\d*[AА]\d[AА]', 6, 'А'),
        (r'\b[0]{1,3}-\d{1,4}[AА]\d*\w*', '100-250', 'А'),
        (r'^\d-\d[AА]$', '40-100', 'А'),
        (r'^44-100[AА]$', '40-100', 'А'),
        (r'^00кА$', 100, 'кА'),
        (r'^[2-9][1-9][1-9]кА$', 100, 'кА'),
        (r'^[1-9][0OQ]кА$', short_value[0]+'0', 'кА'),
        # (r'^[^0-9-]*(-?\d{4,}[AА])$', 5000, 'А'),
        # (r'^[AА][0-9]', 10, 'А'),
        # (r'^[AА]\d{2,3}', 50, 'А'),
        (r'^[^1]\d{2}кА', short_value[0:-2], 'кА'),
        (r'^\d*[4]$', short_value[:-1], 'А'),
        (r'^0-125[AА]', '50-125', 'А'),
        (r'^\w{1}\d*-\d*[AА]$',short_value[0:-1], 'А'),
        (r'^[AА]{1,3}', 6, 'А'),
        (r'^\d*[AА]{2}',short_value[:-2], 'А'),
        # (r'^\s*00[АаA]\s*$', 1000, 'А')
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

def expand_current_name(short_value):
    rules = [
        (r'HG0\d{2}', 'HGD'+short_value[-2:]),
        (r'HG0\d{2}[A-Z]', 'HGD'+short_value[-3:]),
    ]

    for pattern, multiplier in rules:
        if re.fullmatch(pattern, short_value):
            return f"{multiplier}"
        
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

def search_rcd_or_diff(text):
    text_strip = text.strip()

    # Паттерны для УЗО
    rcd_patterns = [
        r'\b\d+[mM][AА]\b',           # 30mA, 10мА
        r'[CС]?\d+/\d+\.?\d*/\d+',    # 40/0,03/2
        r'\d+\.?\d*\s*[mMмМ]?[AА]',   # 0,03, 0,03А, 0,03мА
        r'УЗО', r'RCD',                # Прямые обозначения
        r'QD\d*', r'QDL\d*',           # Маркировки УЗО
        r'FD\d*', r'FDL\d*',           # Маркировки УЗО
    ]
    
    # Паттерны для диффавтоматов (комбинация характеристик)
    diff_patterns = [
        # Должны быть ОДНОВРЕМЕННО характеристика автомата И дифференциальный ток
        r'[BCDСВ]\d+[AА]?\s*\d+[mMмМ][AА]',  # C20 30мА, B16 30mA
        r'[BCDСВ]\d+[AА]?/\d+[mMмМ][AА]',    # C20/30мА, B16/30mA
        r'QFD\d*', r'QFDL\d*',                # Обозначения диффавтоматов
        r'ABAT32',                            # Маркировки диффавтоматов
        r'[BCDСВ]\d+[AА]?/\d+\.?\d*/\d+',    # C16/0,03/2
        r'\d+[AА]\s*\d+\.?\d*[mMмМ]?[AА]',   # 16А 0,03А, 20А 30мА
        r'^[0-9]{2}0[0-9][AА]$',
    ]

    other_patterns = [
        r'QS\d+',
        r'KM\d+',
        r'QSG\d+',
    ]

    is_rcd = any(re.fullmatch(pattern, text_strip) for pattern in rcd_patterns)
    is_diff = is_definitely_diff(text_strip)
    is_other = any(re.fullmatch(pattern, text_strip) for pattern in other_patterns)
    
    return is_rcd or is_diff or is_other

def is_definitely_diff(text):
    """Строгая проверка на диффавтомат (должны быть ОБЕ характеристики)"""
    text_upper = text.upper()
    
    # Явные обозначения диффавтоматов (полное совпадение)
    explicit_diff = re.search(r'QFD\d*|QFDL\d*|ABAT32', text_upper)
    if explicit_diff:
        return True
    
    # Должны присутствовать ОДНОВРЕМЕННО:
    # 1. Характеристика автомата (C10, B16, D20 и т.д.) ИЛИ просто ток (16А, 20А)
    # 2. Дифференциальный ток (30мА, 0,03А, 100mA и т.д.)
    
    # Паттерны для характеристики автомата или номинального тока
    breaker_patterns = [
        r'[BCDСВ]\d+[AА]?',  # C20, B16A
        r'\b\d+[AА]\b',       # 16А, 20А
        r'^[0-9]{2}0[0-9][AА]$',
    ]
    
    # Паттерны для дифференциального тока
    diff_current_patterns = [
        r'\d+[mMмМ][AА]',           # 30мА, 100mA
        r'\d+\.?\d*\s*[mMмМ]?[AА]', # 0,03, 0,03А, 0,03мА
        r'\d+/\d+\.?\d*/\d+',       # 40/0,03/2
        r'^[0-9]{2}0[0-9][AА]$',
    ]
    
    has_breaker_char = any(re.fullmatch(pattern, text_upper) for pattern in breaker_patterns)
    has_diff_current = any(re.fullmatch(pattern, text_upper) for pattern in diff_current_patterns)
    
    return has_breaker_char and has_diff_current

def is_ta(text):
    """Простая проверка на трансформаторы по ключевым словам"""
    if not text:
        return False
        
    text_upper = text.upper()
    
    # Просто проверяем наличие ключевых паттернов трансформаторов
    transformer_indicators = [
        r'/5[АA]?\b',      # содержит /5 или /5A
        r'\bTA\d',         # содержит TA1, TA2 и т.д.
        r'\b\d{3,4}/5',    # содержит 300/5, 400/5
    ]
    
    for pattern in transformer_indicators:
        if re.search(pattern, text_upper):
            print(f"✅ Найден трансформатор по паттерну '{pattern}': '{text}'")
            return True
    
    return False

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
    
    if is_ta(text):
        print(f"Найден трансформатор: '{text}'")
        return search_ta(text)
    
   
    if search_rcd_or_diff(text):
        print(f"Отсеян УЗО/диффавтомат: '{text}'")
        return 
        

    id_qf = r"\b\d*[OQ0]F[TG]?\d*\.?\d*\b"  # Идентификатор (например, QF19)
    current_pattern = r"\b\w?\d{1,4}(?:[.,]\d+)?\s*-\s*\d{1,4}(?:[oO0])?(?:[.,]\d+)?\s*[AА]\b|b\d+[oO0Q]?\d?[AА4]\b|\b[BВCС]?\d+[oO0Q]?\d?[AА4]\b|\b[BВCС]\d+\s*\b"
    name_pattern = r"\s*((?:[BВ][АA]\s*\d{1,4}(?:-\d{1,4})?(?:/\d{1,4})?[A-Z]*)(?:-\d{1,2})?|(?:[HН]G[A-Z]?\s*\d{2,4}[A-Z]?)|(?:U[A-Z]{1,3}\d{2,4}[A-Z]?))\s*"
    voltage_pattern = r"([AА][CС]\d+/\d{3}|[AА][CС]\s*\d{3}|\b\d{3}[AА][CС])"
    current_close_pattern = r"\d{2,4}(?:[oO0])?(?=\D*kA)"
    poles_pattern = r"\d{1}[PpПп]"
    character_pattern = r"[CСDG]"
    device_id = ''.join(re.findall(id_qf, text))
    device_name = ''.join(re.findall(name_pattern, text))
    polus = ''.join(re.findall(poles_pattern, text))[:-1] if ''.join(re.findall(poles_pattern, text)) else ''
    mount_type = ''
    current_range = ''.join(re.findall(current_pattern, text)) 
    current_voltage = ''.join(re.findall(voltage_pattern, text))
    if re.findall(voltage_pattern, text) and re.fullmatch(r'[AА][CС]\d+/\d{3}|[AА][CС]\s*\d{3}', current_voltage):
      current_voltage = current_voltage + 'В' 

    current_close = ''.join(re.findall(current_close_pattern, text))[0:] + 'кА' if re.findall(current_close_pattern, text) else ''
    if re.fullmatch(r'\b\d+\s*[AА]?\b', current_range):
        current_range = fix_current(current_range)
        current_range = fix_merged_currents(current_range)
    else:
        current_range = fix_current_value(current_range)
    if (current_range != '' or current_voltage != '' or current_close != ''):
        if current_range == "64А":
            return
        current_range = expand_current_value(current_range)
        current_close = expand_current_value(current_close)
        current_voltage = expand_voltage(current_voltage)
        device_name = expand_current_name(device_name)
        new_qf = qf.create_qf(device_id, device_name, current_range, current_voltage, current_close, polus, mount_type)
        new_qf.print_data()

        return new_qf
    return

def search_ta(text):
    ta_text =  r"[\dТT]*[ТT]?[АA]+[\dТT]*-[\dТT]*[ТT]?[АA]+[\dТT]*|[\dТT]*[ТT]?[АA]+[\dТT]*...[\dТT]*[ТT]?[АA]+[\dТT]*"
    ta_character = r'^\d{1,2}[0OQ]{1,3}/5[AS]?$'
    ta_exc = ''.join(re.findall(ta_text, text))
    if (ta_exc != ''):
        ta_exc = ta_current_value(ta_exc)
        new_ta = ta.create_ta(ta_exc)
        return new_ta
    return

# if __name__ == "__main__":
#     test_cases = [
#     # Корректные диапазоны
#     "1ТА1-1ТА3",
#     "2ТА4-2ТА6", 
#     "3ТА1-3ТА3",
    
#     # Ошибочные случаи из примеров
#     "31А1-31А3",
#     "1ТАТ-11А3", 
#     "ТТАТ-1ТА3",  # Должно стать: 1ТА1 - 1ТА3
#     "TTA1-1TA3",
#     "2TA1-21A3",
    
#     # Одиночные значения
#     "1ТА1", "2ТА4", "31А1", "1ТАТ", "ТТАТ", "11А3",
    
#     # Дополнительные тесты
#     "ТТА1", "Т1АТ", "1ТТ3", "ТА2", "Т3",
#     "100А", "250-630А", "5кА"
#     ]
#     ta_list = []
#     print("Тестирование функции:")
#     for test in test_cases:
#         corrected = ta_current_value(test)
#         print(f"'{test}' -> '{corrected}'")
#         ta_correct = search_ta(test)
#         if ta_correct != None:
#             ta_list.append(ta_correct)
#     for ta_object in ta_list:
#         print(f"Имя:{ta_object.ta_name}, Количество: {ta_object.ta_quantity}")

