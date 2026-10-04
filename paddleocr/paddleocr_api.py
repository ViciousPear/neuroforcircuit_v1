from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
from PIL import Image
import io
import os


os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paddleocr-api")


class PaddleOCRService:
    """
    Сервис PaddleOCR для распознавания текста на изображениях.

    Инкапсулирует:
        - инициализацию PaddleOCR с кастомными путями к моделям
          (det — Multilingual_PP-OCRv4, rec — cyrillic_PP-OCRv4,
          cls — ch_ppocr_mobile_v2.0);
        - настройку логирования (StreamHandler, без propagate);
        - распознавание текста методом ocr(det=False, cls=False)
          с предварительным апскейлом маленьких ROI;
        - заглушку cleanup().

    Атрибуты:
        temp_dir (str | None): Зарезервировано под временную директорию
            (сейчас не используется).
        ocr (PaddleOCR | None): Загруженная модель или None при ошибке.
    """

    def __init__(self):
        """
        Инициализирует сервис: загружает PaddleOCR и настраивает логирование.

        Returns:
            None
        """
        self.temp_dir = None
        self.ocr = None
        self._initialize()
        self._setup_logging()

    def _setup_logging(self):
        """
        Настраивает StreamHandler для логгера 'paddleocr-api'.

        Удаляет все существующие обработчики, добавляет один StreamHandler
        с единым форматом, устанавливает уровень INFO и отключает
        propagate, чтобы сообщения не дублировались в корневом логгере.

        Returns:
            None
        """
        logger = logging.getLogger("paddleocr-api")
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.propagate = False
        logger.info("Custom logging configured successfully")

    def _initialize(self):
        """
        Загружает PaddleOCR с кастомными путями к моделям.

        Параметры:
            - use_doc_orientation_classify=False
            - use_doc_unwarping=False
            - use_textline_orientation=False
            - lang='en'
            - det=True, rec=True, use_angle_cls=False, show_log=False
            - det_model_dir / rec_model_dir / cls_model_dir указывают
              на локальные веса в /app/.paddleocr/whl/.

        При ошибке логирует сообщение и устанавливает self.ocr = None.

        Returns:
            None
        """
        try:
            from paddleocr import PaddleOCR

            # Включаем det=True, чтобы распознавать многострочные ROI
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='en',
                det=True,
                rec=True,
                use_angle_cls=False,
                show_log=False,
                det_model_dir="/app/.paddleocr/whl/det/ml/Multilingual_PP-OCRv4_det_infer",
                rec_model_dir="/app/.paddleocr/whl/rec/cyrillic/cyrillic_PP-OCRv4_rec_infer",
                cls_model_dir="/app/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"
            )
            logger.info("PaddleOCR service initialized with DETECTION (det=True)")

        except Exception as e:
            logger.error(f" PaddleOCR initialization failed: {e}")
            self.ocr = None

    def recognize(self, image_array):
        """
        Основной метод распознавания текста.

        Логика:
            1. Если модель не загружена — возвращает "".
            2. Копирует входное изображение.
            3. Если h < 100 или w < 100 — увеличивает изображение
               (INTER_CUBIC) до минимальной стороны 100 px.
            4. Вызывает self.ocr.ocr(rgb_image, det=False, cls=False)
               (быстрый режим без детекции).
            5. Парсит результат: ожидает список вида
               [[(text, conf), ...]] и собирает непустые тексты.
            6. Склеивает тексты через пробел.

        При любом исключении логирует ошибку и traceback, возвращает "".

        Args:
            image_array (np.ndarray): Входное изображение (RGB).

        Returns:
            str: Распознанный текст или "" при ошибке/отсутствии текста.
        """
        if self.ocr is None:
            return ""

        try:
            rgb_image = image_array.copy()

            h, w = rgb_image.shape[:2]
            if h < 100 or w < 100:
                scale = max(100 / h, 100 / w)
                rgb_image = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            logger.info(" Running FAST PaddleOCR recognition with ocr(det=False)...")

            result = self.ocr.ocr(rgb_image, det=False, cls=False)

            texts = []
            if result and isinstance(result, list):
                inner = result[0] if isinstance(result[0], list) else result
                for item in inner:
                    if isinstance(item, tuple) and len(item) == 2:
                        text_str, conf = item
                        if text_str and str(text_str).strip():
                            texts.append(str(text_str).strip())

            if texts:
                final_text = ' '.join(texts)
                logger.info(f"PaddleOCR result: '{final_text}'")
                return final_text
            else:
                logger.warning("No text recognized by PaddleOCR")
                return ""

        except Exception as e:
            logger.error(f"PaddleOCR recognition error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def cleanup(self):
        """
        Заглушка очистки ресурсов.

        В текущей реализации ничего не делает (модель PaddleOCR
        не требует явного освобождения на стороне сервиса).

        Returns:
            None
        """
        pass


# Глобальная ссылка на сервис - заполняется в lifespan
paddle_service: PaddleOCRService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом PaddleOCR-сервиса (startup / shutdown).

    Заменяет устаревшие @app.on_event("startup") и
    @app.on_event("shutdown"). На старте создает глобальный
    paddle_service и загружает модель; на завершении вызывает
    paddle_service.cleanup().

    Args:
        app (FastAPI): Экземпляр приложения (передается FastAPI
            автоматически).

    Yields:
        None: Управление передается приложению между startup и shutdown.
    """
    global paddle_service

    # STARTUP
    logger.info("Starting PaddleOCR service...")
    paddle_service = PaddleOCRService()
    if paddle_service.ocr is not None:
        logger.info("PaddleOCR service is ready")
    else:
        logger.error("PaddleOCR service failed to initialize")

    try:
        yield
    finally:
        # SHUTDOWN 
        logger.info("Shutting down PaddleOCR service...")
        if paddle_service is not None:
            paddle_service.cleanup()
            paddle_service = None
        logger.info("PaddleOCR service stopped")


app = FastAPI(
    title="PaddleOCR API",
    version="2.0.0",
    lifespan=lifespan
)


def enhance_image(image_array_rgb):
    """
    Улучшает качество изображения (принимает RGB, возвращает RGB).

    Алгоритм:
        1. RGB -> BGR (для работы OpenCV CLAHE).
        2. BGR -> LAB.
        3. CLAHE (clipLimit=3.0, tileGridSize=(8, 8)) к L-каналу.
        4. LAB -> BGR.
        5. Повышение резкости через filter2D с ядром
           [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]].
        6. BGR -> RGB (для корректного OCR).

    При ошибке логирует предупреждение и возвращает исходное
    RGB-изображение без изменений.

    Args:
        image_array_rgb (np.ndarray): Входное изображение (RGB).

    Returns:
        np.ndarray: Улучшенное изображение (RGB).
    """
    try:
        # Перевод в BGR для работы OpenCV CLAHE
        image_array_bgr = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)

        lab = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_bgr = cv2.filter2D(enhanced_bgr, -1, kernel)

        # Возвращение обратно в RGB для корректного OCR
        return cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning(f"Failed to enhance image, using raw: {e}")
        return image_array_rgb


def smart_filter(text, whitelist):
    """
    Умная фильтрация текста OCR с поддержкой кириллицы.

    Логика:
        - Разбивает текст на слова.
        - Для каждого слова оставляет только символы из whitelist.
        - Слово сохраняется, если выполнено хотя бы одно условие:
            * is_component: содержит букву из 'RCLQDU' (в верхнем
              регистре) и цифру;
            * is_value: содержит цифру и длина >= 2;
            * is_cyrillic: содержит хотя бы один кириллический символ;
            * is_valid: длина >= 2.
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

        is_component = any(c in 'RCLQDU' for c in filtered.upper()) and any(c.isdigit() for c in filtered)
        is_value = any(c.isdigit() for c in filtered) and len(filtered) >= 2
        is_cyrillic = any('А' <= c <= 'я' or c in 'Ёё' for c in filtered)
        is_valid = len(filtered) >= 2

        if (is_component or is_value or is_cyrillic or is_valid) and filtered:
            filtered_words.append(filtered)

    return ' '.join(filtered_words)


@app.post("/recognize")
async def recognize_text(
    file: UploadFile = File(...),
    whitelist: str = "ABCDFGOQSPpHTMmkU0123456789/.,-",
    enhance: bool = True
):
    """
    Основной эндпоинт распознавания текста.

    Пайплайн:
        1. Проверяет, что PaddleOCR-сервис загружен (иначе 500).
        2. Читает файл, декодирует в PIL Image → RGB → np.ndarray.
        3. Если enhance=True — применяет enhance_image.
        4. Вызывает paddle_service.ocr.ocr(processed_array, det=True,
           cls=False) — режим с детекцией (многострочные ROI).
        5. Безопасно парсит многострочную структуру результата:
           для каждого box_info извлекает (text, conf), собирает
           raw_lines и filtered_lines (smart_filter по каждой строке).
        6. Возвращает JSON с полями:
            - "text": отфильтрованный многострочный текст;
            - "raw_text": сырой многострочный текст;
            - "engine": "paddleocr".

    Args:
        file (UploadFile): Изображение ROI с текстом.
        whitelist (str): Строка допустимых символов для smart_filter.
            По умолчанию "ABCDFGOQSPpHTMmkU0123456789/.,-".
        enhance (bool): Применять ли enhance_image перед OCR.
            По умолчанию True.

    Returns:
        JSONResponse: {
            "text": str,
            "raw_text": str,
            "engine": "paddleocr"
        }

    Raises:
        HTTPException(500): Если PaddleOCR-сервис недоступен или
            возникла ошибка при обработке.
    """
    if paddle_service is None or paddle_service.ocr is None:
        raise HTTPException(500, "PaddleOCR service not available")

    try:
        logger.info(f" Processing {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)

        # 1. Предобработка изображения
        if enhance:
            processed_array = enhance_image(image_array)
        else:
            processed_array = image_array

        # 2. Сырой результат от PaddleOCR (список строк из det=True)
        ocr_result = paddle_service.ocr.ocr(processed_array, det=True, cls=False)

        raw_lines = []
        filtered_lines = []

        # 3. Безопасный парсинг многострочной структуры детектора
        if ocr_result and isinstance(ocr_result, list):
            for line in ocr_result:
                if line is None:
                    continue
                for box_info in line:
                    # Извлечение текста из структуры: [[[x1,y1],...], ("текст", уверенность)]
                    if isinstance(box_info, list) and len(box_info) >= 2:
                        text_data = box_info[1]
                        if isinstance(text_data, tuple) and len(text_data) > 0:
                            raw_text = text_data[0].strip()
                            if raw_text:
                                raw_lines.append(raw_text)
                                # Применение белого списка к каждой найденной строке отдельно
                                filtered_line = smart_filter(raw_text, whitelist)
                                if filtered_line:
                                    filtered_lines.append(filtered_line)

        # 4. Сбор итоговых строк
        final_raw_text = "\n".join(raw_lines)
        final_filtered_text = "\n".join(filtered_lines)

        logger.info(f" PaddleOCR completed (multiline):\n'{final_filtered_text}'")

        return JSONResponse(content={
            "text": final_filtered_text,
            "raw_text": final_raw_text,
            "engine": "paddleocr"
        })

    except Exception as e:
        logger.error(f" PaddleOCR API error: {e}")
        raise HTTPException(500, str(e))


@app.get("/health")
async def health_check():
    """
    Проверка готовности сервиса.

    Returns:
        dict: {
            "status": "ready" | "not_ready",
            "engine": "paddleocr"
        }
    """
    return {
        "status": "ready" if (paddle_service is not None and paddle_service.ocr) else "not_ready",
        "engine": "paddleocr"
    }