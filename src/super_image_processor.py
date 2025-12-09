# super_image_fixed.py
import cv2
import numpy as np
import torch
from super_image import EdsrModel, ImageLoader
from PIL import Image
import gc

class SuperImageFixed:
    def __init__(self, model_name='eugenesiow/edsr-base', scale=2):
        self.model_name = model_name
        self.scale = scale
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация модели"""
        try:
            print(f"🔄 Загрузка {self.model_name}...")
            
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
        Исправляем форму массива из (C,H,W) в (H,W,C)
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
        Обработка изображения по тайлам с исправленной формой
        """
        h, w = image.shape[:2]
        output_h, output_w = h * self.scale, w * self.scale
        is_grayscale = len(image.shape) == 2
        
        # Создаем выходное изображение
        if is_grayscale:
            output = np.zeros((output_h, output_w), dtype=np.uint8)
        else:
            output = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        
        print(f"Обработка {h}x{w} -> {output_h}x{output_w} тайлами {tile_size}x{tile_size}")
        
        # Обрабатываем тайлы
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Вырезаем тайл с padding для границ
                y1, y2 = y, min(y + tile_size, h)
                x1, x2 = x, min(x + tile_size, w)
                
                tile = image[y1:y2, x1:x2]
                
                if tile.size == 0:
                    continue
                
                # Добавляем padding если тайл меньше размера
                pad_bottom = tile_size - (y2 - y1) if y2 - y1 < tile_size else 0
                pad_right = tile_size - (x2 - x1) if x2 - x1 < tile_size else 0
                
                if pad_bottom > 0 or pad_right > 0:
                    if is_grayscale:
                        tile = np.pad(tile, ((0, pad_bottom), (0, pad_right)), mode='reflect')
                    else:
                        tile = np.pad(tile, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')
                
                # Обрабатываем тайл
                enhanced_tile = self._enhance_tile(tile)
                
                if enhanced_tile is not None:
                    # Вычисляем координаты для вставки
                    output_y, output_x = y1 * self.scale, x1 * self.scale
                    tile_h, tile_w = enhanced_tile.shape[:2]
                    
                    # Убираем padding из результата
                    result_h = min(tile_h - pad_bottom * self.scale, (y2 - y1) * self.scale)
                    result_w = min(tile_w - pad_right * self.scale, (x2 - x1) * self.scale)
                    
                    if result_h > 0 and result_w > 0:
                        output_tile = enhanced_tile[:result_h, :result_w]
                        output[output_y:output_y+result_h, output_x:output_x+result_w] = output_tile
                
                # Очищаем память
                gc.collect()
                
                print(f"Тайл [{y},{x}] обработан", end='\r')
        
        print("\n Все тайлы обработаны!")
        return output
    
    def _enhance_tile(self, tile):
        """Обработка одного тайла с исправленной формой"""
        try:
            # Конвертируем в PIL
            if len(tile.shape) == 2:
                pil_tile = Image.fromarray(tile)
                was_grayscale = True
            else:
                pil_tile = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                was_grayscale = False
            
            # Обрабатываем
            inputs = ImageLoader.load_image(pil_tile)
            
            with torch.no_grad():
                preds = self.model(inputs)
            
            # Исправляем форму массива
            enhanced_array = self._correct_array_shape(preds, tile.shape)
            
            # Конвертируем обратно в BGR если нужно
            if not was_grayscale:
                enhanced_array = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
            
            return enhanced_array
                
        except Exception as e:
            print(f"Ошибка обработки тайла: {e}")
            return None
    
    def enhance_direct(self, image):
        """Прямая обработка для маленьких изображений"""
        try:
            # Конвертируем в PIL
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
                was_grayscale = True
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                was_grayscale = False
            
            # Обрабатываем
            inputs = ImageLoader.load_image(pil_image)
            
            with torch.no_grad():
                preds = self.model(inputs)
            
            # Исправляем форму
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
        Улучшение изображения
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована")
        
        if image is None or image.size == 0:
            return image
        
        print(f"Входное изображение: {image.shape}")
        
        try:
            # Определяем оптимальный размер тайла
            h, w = image.shape[:2]
            
            if use_tiles and (h > 512 or w > 512):
                # Автоматический подбор размера тайла
                tile_size = min(128, h // 4, w // 4)
                tile_size = max(64, tile_size)  # Минимум 64 пикселя
                print(f"🔲 Обработка по тайлам {tile_size}x{tile_size}...")
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
    Исправленная функция улучшения
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
        
