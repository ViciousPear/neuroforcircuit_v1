import cv2
import numpy as np
import torch
from super_image import EdsrModel, ImageLoader
from PIL import Image
import gc


class SuperImageFixed:
    """
    Обертка над моделью EDSR (super-image) для увеличения/улучшения
    изображений с поддержкой тайловой обработки.

    Класс инкапсулирует:
        - загрузку предобученной модели EdsrModel (по умолчанию
          'eugenesiow/edsr-base') с заданным scale;
        - прямую обработку небольших изображений (enhance_direct);
        - тайловую обработку больших изображений (process_in_tiles)
          с автоматическим подбором размера тайла и padding'ом;
        - корректировку формы тензора (C,H,W) → (H,W,C) и
          преобразование в uint8.

    Атрибуты:
        model_name (str): Имя предобученной модели.
        scale (int): Коэффициент увеличения (2 или 4).
        model (EdsrModel | None): Загруженная модель или None при ошибке.
    """

    def __init__(self, model_name='eugenesiow/edsr-base', scale=2):
        """
        Инициализирует процессор и загружает модель.

        Args:
            model_name (str): Имя предобученной модели EDSR.
                По умолчанию 'eugenesiow/edsr-base'.
            scale (int): Коэффициент увеличения. По умолчанию 2.

        Returns:
            None
        """
        self.model_name = model_name
        self.scale = scale
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Загружает модель EdsrModel.from_pretrained и переводит ее в eval-режим.

        Перед загрузкой очищает кеш CUDA (если доступен) и вызывает
        gc.collect(). При ошибке логирует сообщение и пробрасывает
        исключение дальше.

        Returns:
            None

        Raises:
            Exception: Любая ошибка при загрузке модели.
        """
        try:
            print(f"Загрузка {self.model_name}...")

            # Очищаем память
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            self.model = EdsrModel.from_pretrained(self.model_name, scale=self.scale)
            self.model.eval()

            print("Модель загружена!")

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            raise

    def _correct_array_shape(self, tensor, original_shape):
        """
        Преобразует тензор (C,H,W) или (1,C,H,W) в numpy-массив (H,W,C).

        Шаги:
            1. Если тензор 4-мерный — squeeze(0).
            2. Перевод в CPU numpy.
            3. Транспонирование (C,H,W) → (H,W,C).
            4. Ограничение значений [0,1] → [0,255] и приведение к uint8.

        Args:
            tensor (torch.Tensor): Выход модели.
            original_shape (tuple): Исходная форма изображения
                (используется только для совместимости, в расчетах
                не участвует).

        Returns:
            np.ndarray: Массив (H,W,C) типа uint8.
        """
        # Убираем batch dimension если есть
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Конвертируем в numpy
        array = tensor.cpu().numpy()

        # Меняем местами оси: (C, H, W) -> (H, W, C)
        array = np.transpose(array, (1, 2, 0))

        # Ограничиваем значения и конвертируем в uint8
        array = np.clip(array * 255, 0, 255).astype(np.uint8)

        return array

    def process_in_tiles(self, image, tile_size=128):
        """
        Обрабатывает изображение по тайлам заданного размера.

        Логика:
            - Создает выходной массив размером (h*scale, w*scale)
              (grayscale или BGR в зависимости от входного изображения).
            - Идет по изображению с шагом tile_size.
            - Для каждого тайла:
                * вырезает фрагмент [y1:y2, x1:x2];
                * если тайл меньше tile_size, добавляет reflect-padding;
                * обрабатывает через _enhance_tile;
                * вставляет результат в выходной массив с учетом
                  удаления padding'а;
                * вызывает gc.collect().
            - Печатает прогресс (последний обработанный тайл).

        Args:
            image (np.ndarray): Входное изображение (BGR или grayscale).
            tile_size (int): Размер тайла. По умолчанию 128.

        Returns:
            np.ndarray: Увеличенное изображение того же типа, что и вход
                (grayscale или BGR).
        """
        h, w = image.shape[:2]
        output_h, output_w = h * self.scale, w * self.scale
        is_grayscale = len(image.shape) == 2

        # Создание выходного изображения
        if is_grayscale:
            output = np.zeros((output_h, output_w), dtype=np.uint8)
        else:
            output = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        print(f"Обработка {h}x{w} -> {output_h}x{output_w} тайлами {tile_size}x{tile_size}")

        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Вырезка тайла с padding для границ
                y1, y2 = y, min(y + tile_size, h)
                x1, x2 = x, min(x + tile_size, w)

                tile = image[y1:y2, x1:x2]

                if tile.size == 0:
                    continue

                # Добавление padding если тайл меньше размера
                pad_bottom = tile_size - (y2 - y1) if y2 - y1 < tile_size else 0
                pad_right = tile_size - (x2 - x1) if x2 - x1 < tile_size else 0

                if pad_bottom > 0 or pad_right > 0:
                    if is_grayscale:
                        tile = np.pad(tile, ((0, pad_bottom), (0, pad_right)), mode='reflect')
                    else:
                        tile = np.pad(tile, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')

                # Обработка твйла
                enhanced_tile = self._enhance_tile(tile)

                if enhanced_tile is not None:
                    # Вычисление координат для вставки
                    output_y, output_x = y1 * self.scale, x1 * self.scale
                    tile_h, tile_w = enhanced_tile.shape[:2]

                    # Удаление padding из результата
                    result_h = min(tile_h - pad_bottom * self.scale, (y2 - y1) * self.scale)
                    result_w = min(tile_w - pad_right * self.scale, (x2 - x1) * self.scale)

                    if result_h > 0 and result_w > 0:
                        output_tile = enhanced_tile[:result_h, :result_w]
                        output[output_y:output_y+result_h, output_x:output_x+result_w] = output_tile

                # Очистка памяти
                gc.collect()

                print(f"Тайл [{y},{x}] обработан", end='\r')

        print("\n Все тайлы обработаны!")
        return output

    def _enhance_tile(self, tile):
        """
        Обрабатывает один тайл через модель EDSR.

        Конвертирует тайл в PIL (grayscale или RGB), загружает через
        ImageLoader.load_image, прогоняет через модель под torch.no_grad(),
        корректирует форму (_correct_array_shape) и при необходимости
        возвращает BGR.

        Args:
            tile (np.ndarray): Тайл изображения (BGR или grayscale).

        Returns:
            np.ndarray | None: Увеличенный тайл (BGR или grayscale)
                или None при ошибке.
        """
        try:
            # Конвертация в PIL
            if len(tile.shape) == 2:
                pil_tile = Image.fromarray(tile)
                was_grayscale = True
            else:
                pil_tile = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                was_grayscale = False

            inputs = ImageLoader.load_image(pil_tile)

            with torch.no_grad():
                preds = self.model(inputs)

            enhanced_array = self._correct_array_shape(preds, tile.shape)

            if not was_grayscale:
                enhanced_array = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)

            return enhanced_array

        except Exception as e:
            print(f"Ошибка обработки тайла: {e}")
            return None

    def enhance_direct(self, image):
        """
        Прямая обработка изображения (без тайлов) для небольших картинок.

        Конвертирует изображение в PIL, загружает через ImageLoader,
        прогоняет через модель под torch.no_grad(), корректирует форму
        и при необходимости возвращает BGR.

        Args:
            image (np.ndarray): Входное изображение (BGR или grayscale).

        Returns:
            np.ndarray | None: Увеличенное изображение (BGR или grayscale)
                или None при ошибке.
        """
        try:
            # Конвертация в PIL
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
                was_grayscale = True
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                was_grayscale = False

            inputs = ImageLoader.load_image(pil_image)

            with torch.no_grad():
                preds = self.model(inputs)

            enhanced_array = self._correct_array_shape(preds, image.shape)

            if was_grayscale:
                return enhanced_array
            else:
                return cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"Ошибка прямой обработки: {e}")
            return None

    def enhance(self, image, use_tiles=True):
        """
        Улучшает изображение: тайлами (для больших) или напрямую
        (для маленьких).

        Логика выбора режима:
            - Если use_tiles=True и max(h, w) > 512 — тайловая обработка
              с tile_size = max(64, min(128, h // 4, w // 4)).
            - Иначе — enhance_direct.

        Args:
            image (np.ndarray): Входное изображение (BGR или grayscale).
            use_tiles (bool): Разрешить тайловую обработку. По умолчанию True.

        Returns:
            np.ndarray | None: Улучшенное изображение или None при ошибке.

        Raises:
            RuntimeError: Если модель не инициализирована.
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована")

        if image is None or image.size == 0:
            return image

        print(f"Входное изображение: {image.shape}")

        try:
            h, w = image.shape[:2]

            if use_tiles and (h > 512 or w > 512):
                # Автоматический подбор размера тайла
                tile_size = min(128, h // 4, w // 4)
                tile_size = max(64, tile_size)  # Минимум 64 пикселя
                print(f"Обработка по тайлам {tile_size}x{tile_size}...")
                return self.process_in_tiles(image, tile_size=tile_size)
            else:
                print("Прямая обработка...")
                return self.enhance_direct(image)

        except Exception as e:
            print(f"Ошибка улучшения: {e}")
            return None


# Глобальный экземпляр
_super_image_fixed = None


def get_super_image_fixed():
    """
    Возвращает глобальный экземпляр SuperImageFixed (ленивая инициализация).

    При первом вызове создает SuperImageFixed(scale=2) и сохраняет
    в модульной переменной _super_image_fixed. При ошибке инициализации
    логирует сообщение и возвращает None.

    Returns:
        SuperImageFixed | None: Готовый процессор или None при ошибке.
    """
    global _super_image_fixed
    if _super_image_fixed is None:
        try:
            _super_image_fixed = SuperImageFixed(scale=2)
        except Exception as e:
            print(f"Не удалось инициализировать Super-Image: {e}")
            _super_image_fixed = None
    return _super_image_fixed


def enhance_with_super_image(image):
    """
    Удобная обертка: улучшает изображение через глобальный
    SuperImageFixed с тайловой обработкой.

    Если процессор недоступен (get_super_image_fixed вернул None)
    или enhance вернул None — возвращает исходное изображение.
    При исключении логирует ошибку и также возвращает исходное
    изображение.

    Args:
        image (np.ndarray): Входное изображение (BGR или grayscale).

    Returns:
        np.ndarray: Улучшенное изображение или исходное при ошибке.
    """
    processor = get_super_image_fixed()

    if processor is None:
        print("Super-Image недоступен")
        return image

    try:
        result = processor.enhance(image, use_tiles=True)
        if result is not None:
            return result
        else:
            print("Super-Image вернул None")
            return image
    except Exception as e:
        print(f"Super-Image ошибка: {e}")
        return image