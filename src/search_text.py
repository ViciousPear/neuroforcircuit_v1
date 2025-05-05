import numpy
import cv2
import pytesseract
import re
from . import qf, ta

def cleaned_image(resized_image):
    """cleaned_image
    Считывает: изображение с измененным масштабом
    Возращает: улучшенное изображение
    Функция улучшает качество текста изображения 
    с помощью окраски в серый, бинарирования, 
    морфологического расширения и морфологического 
    сужения, что также улучшает качество текста
    """
    # Преобразование в grayscale
    gray_image = cv2.cvtColor(resized_image, 
    cv2.COLOR_BGR2GRAY)

    # Бинарирование
    _, binary_image = cv2.threshold(gray_image, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Создание структурного элемента (ядро для морфологических операций)
    kernel = numpy.ones((3, 3), numpy.uint8)

    # Морфологическое расширение (dilation)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Морфологическое сужение (erosion)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    return eroded_image

def recognize_text_from_bbox(image, x1, y1, x2, y2):
    """ recognize_text_from_bbox
    Считывает текст с куска текста, который отмечен координатами
    Возвращает: распознанный текст без лишних проблелов
    Тип: изображения, числовой текстовый
    Принимается кусок изображения по координатам, увеличивает масштаб
    Для лучшего распознавания вызывает функцию cleaned_image 
    для улучшения качества изображения
    После улучшения качества изображения начинается этап распознавания текста 
    с помощью tesseract OCR
    В настройках распознавания написаны символы, 
    которые должны распозваться и указывается версия для распознавания
    """
    #получение изображений по координатам
    roi = image[y1:y2, x1:x2] 

    #увеличение на 200%
    scale_percent = 200  
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(roi, (width, height), interpolation=cv2.INTER_CUBIC)
   
    eroded_image = cleaned_image(resized_image)

    text = pytesseract.image_to_string(eroded_image, config='--oem 3 -c tessedit_char_whitelist="ABCFGQTmk0123456789/-Азк')  # Запуск OCR

    return text.strip()

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
    ta_text =  r"\d*[ТT][АA]+[A-Za-z]?\d+-\d*[ТT][АA]+[A-Za-z]?\d+"
    ta_exc = ''.join(re.findall(ta_text, text))
    if (ta_exc != ''):
        new_ta = ta.create_ta(ta_exc)
        return new_ta
    return