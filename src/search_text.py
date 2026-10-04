import numpy
import cv2
import base64
import requests
import os
import numpy as np


# local: http://localhost:5003
# production: http://super-image-service:5003
# for tests: http://номер_хоста:5003
class SuperImageClient:
    """
    Клиент для обращения к Super-Image API (улучшение изображений).

    Используется для повышения качества входного изображения перед
    дальнейшей обработкой (детекцией/распознаванием). URL API можно
    передать явно или задать через переменную окружения SUPER_IMAGE_URL.

    Attributes:
        api_url (str): Базовый URL сервиса улучшения изображений.
    """

    def __init__(self, api_url="http://super-image-service:5003"):
        """
        Инициализирует клиент Super-Image API.

        Если api_url не передан (None или пустая строка), значение берется
        из переменной окружения SUPER_IMAGE_URL; если и она не задана —
        используется значение по умолчанию.

        Args:
            api_url (str | None): Базовый URL сервиса. По умолчанию
                "http://super-image-service:5003".
        """
        self.api_url = api_url or os.getenv("SUPER_IMAGE_URL", "http://super-image-service:5003")

    def enhance_image(self, image, use_tiles=True):
        """
        Улучшает изображение через Super-Image API.

        Изображение кодируется в PNG, отправляется POST-запросом на
        {api_url}/enhance с параметрами use_tiles и return_base64=True.
        Ответ содержит base64-encoded PNG, который декодируется обратно
        в изображение OpenCV (BGR).

        Args:
            image (np.ndarray): Входное изображение (OpenCV BGR).
            use_tiles (bool): Использовать ли тайловую обработку на сервере.
                По умолчанию True.

        Returns:
            np.ndarray: Улучшенное изображение (OpenCV BGR). Если запрос
                не удался (RequestException), возвращается исходное
                изображение без изменений.

        Raises:
            ValueError: Если не удалось закодировать изображение в PNG.
            Exception: Если сервер вернул success=False.
        """
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
            return image


def ensure_max_side(image, max_side=2900):
    """
    Гарантирует, что максимальная сторона изображения не превышает лимит.

    Если изображение уже удовлетворяет ограничению — возвращается как есть.
    Иначе пропорционально уменьшается с интерполяцией INTER_AREA
    (качественно для уменьшения).

    Args:
        image (np.ndarray | None): Входное изображение.
        max_side (int): Максимально допустимый размер стороны в пикселях.
            По умолчанию 2900.

    Returns:
        np.ndarray | None: Изображение с максимальной стороной <= max_side,
            либо None, если на вход передан None.
    """
    if image is None:
        return None

    h, w = image.shape[:2]
    current_max = max(h, w)

    if current_max <= max_side:
        return image

    scale = max_side / current_max
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"Изображение слишком большое. Сжато до: {new_w}x{new_h}")

    return resized


def preprocess_crop_for_ocr(crop):
    """
    Оптимизированная предобработка кропа для OCR без уничтожения букв.

    Логика:
        - Если кроп маленький (высота или ширина < 80 px), он увеличивается
          в 2.5–3 раза (INTER_CUBIC), чтобы текст был читаемым для OCR.
        - Перевод в Grayscale.
        - Мягкое адаптивное выравнивание гистограммы (CLAHE,
          clipLimit=2.0, tileGridSize=(8, 8)) вместо жесткого OTSU.
          Это сохраняет тонкие штрихи букв.

    Args:
        crop (np.ndarray | None): Вырезанный фрагмент изображения (BGR).

    Returns:
        np.ndarray | None: Обработанный grayscale-кроп, либо None,
            если на вход передан None или пустой массив.
    """
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]

    if h < 80 or w < 80:
        scale = max(2.5, 100.0 / min(h, w))
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def cleaned_image(resized_image):
    """
    Улучшает качество текста на изображении классическими методами CV.

    Последовательность:
        1. Перевод в Grayscale.
        2. Бинаризация методом OTSU (THRESH_BINARY + THRESH_OTSU).
        3. Морфологическое расширение (dilate) ядром 3x3, 1 итерация.
        4. Морфологическое сужение (erode) тем же ядром, 1 итерация.

    Такая связка (dilate → erode) сглаживает шум и делает штрихи
    текста более четкими.

    Args:
        resized_image (np.ndarray): Изображение (BGR), уже приведенное
            к нужному масштабу.

    Returns:
        np.ndarray: Бинаризованное изображение после морфологических
            операций (одноканальное, uint8).
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
    """
    Распознает текст в заданном bounding box через OCR-сервис.

    Пайплайн:
        1. Вырезает ROI по координатам (x1, y1, x2, y2).
        2. Ограничивает максимальную сторону ROI (ensure_max_side).
        3. Предобрабатывает кроп для OCR (preprocess_crop_for_ocr).
        4. Кодирует результат в PNG и отправляет POST-запросом на
           {api_url}/recognize с whitelist символов и enhance=false.
        5. Возвращает распознанный текст без лишних пробелов.

    URL API берется из аргумента api_url, иначе из переменной окружения
    TESSERACT_API_URL, иначе — значение по умолчанию.

    Args:
        image (np.ndarray): Исходное изображение (OpenCV BGR).
        x1 (int): Левая граница ROI.
        y1 (int): Верхняя граница ROI.
        x2 (int): Правая граница ROI.
        y2 (int): Нижняя граница ROI.
        api_url (str | None): Базовый URL OCR-сервиса. Если None —
            берется из TESSERACT_API_URL или значение по умолчанию.

    Returns:
        str: Распознанный текст (strip). Пустая строка, если запрос
            не удался (RequestException).
    """
    api_url = api_url or os.getenv("TESSERACT_API_URL", "http://paddleocr-service:5000")
    # для контейнера: "http://paddleocr-service:5000"
    # для тестов "http://82.202.129.245:5000"
    # для внесения изменений "http://localhost:5000"
    # получение изображений по координатам

    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    print(f"Вырезанный ROI оригинальный: {w}x{h}")

    roi = ensure_max_side(roi)

    # очистка и улучшение качества
    processed_image = preprocess_crop_for_ocr(roi)

    _, buffer = cv2.imencode(".png", processed_image)
    img_bytes = buffer.tobytes()

    try:
        response = requests.post(
            f"{api_url}/recognize",
            files={"file": ("roi.png", img_bytes, "image/png")},
            data={
                "whitelist": "ABCDFGQHTSМmk0123456789/-АСзкМмuFmH",
                "enhance": "false"
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