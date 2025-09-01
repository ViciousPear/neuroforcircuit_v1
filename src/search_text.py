import numpy
import cv2
import base64
import requests
import os

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

def recognize_text_from_bbox(image, x1, y1, x2, y2, api_url=None):
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
    api_url = api_url or os.getenv("TESSERACT_API_URL", "http://82.202.129.245:5000")
    #получение изображений по координатам
    roi = image[y1:y2, x1:x2] 

    #увеличение на 200%
    scale_percent = 200  
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(roi, (width, height), interpolation=cv2.INTER_CUBIC)
   
    eroded_image = cleaned_image(resized_image)
    _, buffer = cv2.imencode(".png", eroded_image)  # Кодируем в PNG
    img_base64 = base64.b64encode(buffer).decode('utf-8')  # Получаем base64

    try:
        response = requests.post(
            f"{api_url}/recognize",
            json={
                "image_base64": img_base64,
                "whitelist": "ABCFGQTmk0123456789/-Азк",
            },
            timeout=300
        )
        response.raise_for_status()
        response_json = response.json()
        text = response_json["text"]
        return text.strip()
    
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса к Tesseract-API: {e}")
        return ""

