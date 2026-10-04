# containers/api/ocr_service.py
import cv2
import numpy as np
import logging
import tempfile
import os
import shutil
import re
from typing import List

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = '1'
os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "2")
os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", "2")

logger = logging.getLogger("ocr-service")


class PaddleOCRService:
    """
    Обертка над PaddleOCR для распознавания текста на изображениях.

    Класс инкапсулирует:
        - ленивую инициализацию PaddleOCR-модели (lang='en', без
          doc-orientation/unwarping/textline-orientation);
        - предобработку изображения (RGB + ограничение максимальной
          стороны);
        - распознавание текста методом ocr() и парсинг результата
          в список строк;
        - очистку временной директории (cleanup).

    Атрибуты:
        temp_dir (str | None): Временная директория для PaddleOCR.
        ocr (PaddleOCR | None): Загруженная модель или None при ошибке.
        max_image_side (int): Максимальная сторона входного изображения.
    """

    def __init__(self, max_image_side: int = 4000):
        """
        Инициализирует сервис и загружает PaddleOCR-модель.

        Создает временную директорию, настраивает переменные окружения
        (USERPROFILE), импортирует PaddleOCR и создает экземпляр модели
        с отключенными doc-orientation/unwarping/textline-orientation
        и lang='en'. При ошибке загрузки self.ocr = None (сервис
        останется работоспособным, но recognize будет возвращать "").

        Args:
            max_image_side (int): Максимальная сторона изображения,
                до которой оно будет уменьшено перед распознаванием.
                По умолчанию 4000.
        """
        self.temp_dir = None
        self.ocr = None
        self.max_image_side = max_image_side
        self._initialize()
        self._setup_logging()

    def _setup_logging(self):
        """
        Настраивает StreamHandler для логгера 'ocr-service'.

        Если у логгера еще нет обработчиков, добавляет один StreamHandler
        с единым форматом и уровнем INFO, отключает propagate, чтобы
        сообщения не дублировались в корневом логгере.

        Returns:
            None
        """
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        logger.info("OCR Service logging configured")

    def _initialize(self):
        """
        Создает временную директорию и загружает PaddleOCR-модель.

        При успехе self.ocr — экземпляр PaddleOCR, при ошибке — None
        (исключение логируется, но не пробрасывается, чтобы сервис
        мог стартовать даже без модели).

        Returns:
            None
        """
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="paddle_ocr_")
            os.environ['USERPROFILE'] = self.temp_dir

            from paddleocr import PaddleOCR

            logger.info("Loading PaddleOCR model...")
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='en'
            )
            logger.info("PaddleOCR model loaded successfully")
        except Exception as e:
            logger.error(f"PaddleOCR initialization failed: {e}")
            self.ocr = None

    def parse_results(self, result) -> List[str]:
        """
        Парсит результат PaddleOCR (метод ocr()) в список строк.

        Ожидаемая структура результата:
            [[[bbox, (text, confidence)], ...]]

        Если основной разбор не дал текстов, используется запасной
        вариант — извлечение строк через регулярное выражение по
        строковому представлению результата (с фильтрацией служебных
        ключей bbox/text/confidence/rec_texts/rec_scores).

        Args:
            result: Сырой результат self.ocr.ocr(...).

        Returns:
            List[str]: Список распознанных текстов без пустых строк.
        """
        if not result:
            return []

        texts = []

        try:
            # Для метода ocr() результат - список списков
            if isinstance(result, list) and len(result) > 0:
                # Первый элемент - результаты для одного изображения
                ocr_result = result[0] if isinstance(result[0], list) else result

                for item in ocr_result:
                    if isinstance(item, list) and len(item) >= 2:
                        # item[0] - bbox, item[1] - (text, confidence)
                        text_data = item[1]
                        if isinstance(text_data, (list, tuple)) and len(text_data) >= 1:
                            text = str(text_data[0]).strip()
                            if text:
                                texts.append(text)
                                confidence = text_data[1] if len(text_data) > 1 else 0
                                logger.debug(f"Recognized: '{text}' ({confidence:.2f})")

            # Запасной вариант: поиск через регулярки в строковом представлении
            if not texts:
                result_str = str(result)
                words = re.findall(r"'([^']+)'", result_str)
                for word in words:
                    if word and len(word) >= 2 and not word.startswith('['):
                        # Фильтруем служебные слова
                        if word not in ['bbox', 'text', 'confidence', 'rec_texts', 'rec_scores']:
                            texts.append(word)

            if texts:
                logger.info(f"Parsed {len(texts)} text items: {texts[:5]}")

        except Exception as e:
            logger.error(f"Error parsing results: {e}")

        return texts

    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Предобрабатывает изображение перед распознаванием.

        Конвертирует BGR → RGB и, если максимальная сторона превышает
        self.max_image_side, пропорционально уменьшает изображение
        с интерполяцией INTER_AREA.

        Args:
            image_array (np.ndarray): Входное изображение (OpenCV BGR).

        Returns:
            np.ndarray: RGB-изображение, при необходимости уменьшенное.
        """
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        h, w = rgb_image.shape[:2]
        if max(h, w) > self.max_image_side:
            scale = self.max_image_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized from {w}x{h} to {new_w}x{new_h}")

        return rgb_image

    def recognize(self, image_array: np.ndarray) -> str:
        """
        Распознает текст на изображении через PaddleOCR.

        Пайплайн:
            1. Если модель не загружена — возвращает "".
            2. Предобрабатывает изображение (preprocess_image).
            3. Вызывает self.ocr.ocr(rgb_image).
            4. Парсит результат (parse_results) и склеивает тексты
               через пробел.

        При любом исключении логирует ошибку и traceback, возвращает "".

        Args:
            image_array (np.ndarray): Входное изображение (OpenCV BGR).

        Returns:
            str: Распознанный текст (склеенные фрагменты) или "" при
                отсутствии текста/ошибке.
        """
        if self.ocr is None:
            return ""

        try:
            rgb_image = self.preprocess_image(image_array)
            logger.info("Running PaddleOCR recognition with ocr()...")

            # Используем ocr() вместо predict()
            result = self.ocr.ocr(rgb_image)

            texts = self.parse_results(result)

            if texts:
                final_text = ' '.join(texts)
                logger.info(f"PaddleOCR result: '{final_text}'")
                return final_text
            else:
                logger.warning("No text recognized")
                return ""

        except Exception as e:
            logger.error(f"Recognition error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def cleanup(self):
        """
        Удаляет временную директорию, созданную при инициализации.

        Использует shutil.rmtree(ignore_errors=True), поэтому ошибки
        удаления не пробрасываются.

        Returns:
            None
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def enhance_image(image_array: np.ndarray) -> np.ndarray:
    """
    Улучшает качество изображения для последующего OCR.

    Алгоритм:
        1. Перевод BGR → LAB.
        2. Применение CLAHE (clipLimit=3.0, tileGridSize=(8, 8))
           к L-каналу.
        3. Обратный перевод LAB → BGR.
        4. Повышение резкости через filter2D с ядром
           [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]].

    При любой ошибке возвращает исходное изображение без изменений.

    Args:
        image_array (np.ndarray): Входное изображение (OpenCV BGR).

    Returns:
        np.ndarray: Улучшенное изображение (OpenCV BGR).
    """
    try:
        lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(enhanced, -1, kernel)
    except:
        return image_array


def smart_filter(text: str, whitelist: str) -> str:
    """
    Фильтрует текст OCR по белому списку символов.

    Логика:
        - Разбивает текст на слова.
        - Для каждого слова оставляет только символы из whitelist.
        - Слово сохраняется, если выполнено хотя бы одно условие:
            is_component: содержит букву из 'RCLQDU' и цифру;
            is_value: содержит цифру и длина >= 2;
            is_valid: длина >= 2.
        - Слова склеиваются через пробел.

    Если text или whitelist пусты — возвращает исходный text.

    Args:
        text (str): Сырой текст от OCR.
        whitelist (str): Строка допустимых символов.

    Returns:
        str: Отфильтрованный текст.
    """
    if not text or not whitelist:
        return text

    allowed = set(whitelist)
    words = text.split()
    filtered_words = []

    for word in words:
        filtered = ''.join(c for c in word if c in allowed)
        is_component = any(c in 'RCLQDU' for c in filtered) and any(c.isdigit() for c in filtered)
        is_value = any(c.isdigit() for c in filtered) and len(filtered) >= 2
        is_valid = len(filtered) >= 2

        if (is_component or is_value or is_valid) and filtered:
            filtered_words.append(filtered)

    return ' '.join(filtered_words)


def resize_image_if_needed(image_array: np.ndarray, max_side: int = 3900) -> np.ndarray:
    """
    Уменьшает изображение, если его максимальная сторона превышает лимит.

    Если max(h, w) <= max_side — возвращает изображение как есть.
    Иначе пропорционально уменьшает с интерполяцией INTER_AREA.

    Args:
        image_array (np.ndarray): Входное изображение.
        max_side (int): Максимально допустимая сторона. По умолчанию 3900.

    Returns:
        np.ndarray: Изображение с максимальной стороной <= max_side.
    """
    h, w = image_array.shape[:2]
    if max(h, w) <= max_side:
        return image_array
    scale = max_side / max(h, w)
    return cv2.resize(image_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)