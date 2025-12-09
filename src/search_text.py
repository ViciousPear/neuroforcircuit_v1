import numpy
import cv2
import base64
import requests
import os
import numpy as np

# local: http://localhost:5003
# production http://super-image-service:5003
class SuperImageClient:
    def __init__(self, api_url="http://super-image-service:5003"):
        self.api_url = api_url or os.getenv("SUPER_IMAGE_URL", "http://super-image-service:5003")
    
    def enhance_image(self, image, use_tiles=True):
        """Улучшение изображения через API"""
        # Конвертируем изображение в bytes
        success, encoded_image = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Не удалось закодировать изображение")
        
        image_bytes = encoded_image.tobytes()
        
        # Отправляем запрос
        files = {'file': ('image.png', image_bytes, 'image/png')}
        params = {'use_tiles': use_tiles, 'return_base64': True}
        
        try:
            response = requests.post(
                f"{self.api_url}/enhance",
                files=files,
                data=params,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                # Декодируем base64 обратно в изображение
                img_data = base64.b64decode(result['image_base64'])
                nparr = np.frombuffer(img_data, np.uint8)
                enhanced_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return enhanced_img
            else:
                raise Exception("Ошибка обработки на сервере")
                
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса к Super-Image API: {e}")
            return None
        
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
    api_url = api_url or os.getenv("TESSERACT_API_URL", "http://paddleocr-service:5000")
    # для контейнера: "http://paddleocr-service:5000"
    # для тестов "http://82.202.129.245:5000"
    # для внесения изменений "http://localhost:5000"
    #получение изображений по координатам
    roi = image[y1:y2, x1:x2] 

    print(f"Размер ROI: {x2-x1} x {y2-y1}")
    
    roi = image[y1:y2, x1:x2]
    print(f"Вырезанный ROI: {roi.shape}")

    # enhance_image = super_image_processor.enhance_with_super_image(roi)
    # super_image_client = SuperImageClient(api_url="http://super-image-service:5003")
    super_image_client = SuperImageClient()
    enhance_image = super_image_client.enhance_image(roi)
    # gray_image = cv2.cvtColor(enhance_image, 
    # cv2.COLOR_BGR2GRAY)
    #увеличение на 200%
    scale_percent = 50  
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(enhance_image, (width, height), interpolation=cv2.INTER_CUBIC)
   
    eroded_image = cleaned_image(resized_image)
    
    _, buffer = cv2.imencode(".png", eroded_image)  # Кодируем в PNG
    img_base64 = base64.b64encode(buffer).decode('utf-8')  # Получаем base64
    img_bytes = buffer.tobytes()

    try:
        response = requests.post(
            f"{api_url}/recognize",
            files={"file": ("roi.png", img_bytes, "image/png")},
            data={
                "whitelist": "ABCDFGQHTSМmk0123456789/-АСзкМмuFmH",
                "enhance": "true"
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

