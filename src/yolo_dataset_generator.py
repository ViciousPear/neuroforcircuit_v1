import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split

class YOLODatasetGenerator:
    def __init__(self, config):
        self.config = config
        self.augmentor = self._create_augmentations()
        os.makedirs(config['dataset_path'], exist_ok=True)
        
    def _create_augmentations(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=20, p=0.3),
            A.GaussianBlur(p=0.1),
            A.CLAHE(p=0.1),
        ], bbox_params=A.BboxParams(format='yolo'))

    def _save_annotation(self, filename, boxes, labels):
        txt_path = os.path.join(self.config['dataset_path'], 'labels', filename)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, 'w') as f:
            for label, box in zip(labels, boxes):
                f.write(f"{label} {' '.join(map(str, box))}\n")

    def _save_image(self, image, filename):
        img_path = os.path.join(self.config['dataset_path'], 'images', filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, image)

    def process_and_augment(self, image_path, boxes, labels):
        original_image = cv2.imread(image_path)
        
        # Сохраняем оригинальные данные
        base_name = os.path.basename(image_path)
        self._save_image(original_image, f"original_{base_name}")
        self._save_annotation(f"original_{base_name.replace('.png', '.txt')}", boxes, labels)
        
        # Применяем аугментации
        for i in range(self.config['augmentation_factor']):
            augmented = self.augmentor(image=original_image, bboxes=boxes, class_labels=labels)
            
            aug_img = augmented['image']
            aug_boxes = np.array(augmented['bboxes'])
            aug_labels = augmented['class_labels']
            
            self._save_image(aug_img, f"aug_{i}_{base_name}")
            self._save_annotation(f"aug_{i}_{base_name.replace('.png', '.txt')}", aug_boxes, aug_labels)

    def create_dataset_structure(self):
        # Создаем структуру папок YOLO
        structure = [
            'images/train',
            'images/val',
            'labels/train',
            'labels/val'
        ]
        
        for folder in structure:
            os.makedirs(os.path.join(self.config['dataset_path'], folder), exist_ok=True)

    def split_dataset(self, test_size=0.2):
        all_files = [f for f in os.listdir(os.path.join(self.config['dataset_path'], 'images'))]
        train_files, val_files = train_test_split(all_files, test_size=test_size)
        
        # Перемещаем файлы в соответствующие папки
        for file in train_files:
            os.rename(
                os.path.join(self.config['dataset_path'], 'images', file),
                os.path.join(self.config['dataset_path'], 'images/train', file)
            )
            os.rename(
                os.path.join(self.config['dataset_path'], 'labels', file.replace('.png', '.txt')),
                os.path.join(self.config['dataset_path'], 'labels/train', file.replace('.png', '.txt'))
            )

        for file in val_files:
            os.rename(
                os.path.join(self.config['dataset_path'], 'images', file),
                os.path.join(self.config['dataset_path'], 'images/val', file)
            )
            os.rename(
                os.path.join(self.config['dataset_path'], 'labels', file.replace('.png', '.txt')),
                os.path.join(self.config['dataset_path'], 'labels/val', file.replace('.png', '.txt'))
            )

# Пример использования
config = {
    'dataset_path': './yolo_dataset',
    'augmentation_factor': 5,  # Количество аугментированных версий для каждого изображения
    'input_shape': (640, 640),
    'class_names': ['resistor', 'capacitor', 'inductor']  # Ваши классы
}

# Инициализация генератора
generator = YOLODatasetGenerator(config)

# Создание структуры папок
generator.create_dataset_structure()

# Обработка исходных изображений (пример для одного изображения)
image_path = "tests/test_image.png"
boxes = [[0.1, 0.2, 0.05, 0.03]]  # Пример в формате YOLO [x_center, y_center, width, height]
labels = [0]  # Индекс класса

generator.process_and_augment(image_path, boxes, labels)

# Разделение на тренировочный и валидационный наборы
generator.split_dataset(test_size=0.2)

# Создание data.yaml
data_yaml = f"""
train: {os.path.join(config['dataset_path'], 'images/train')}
val: {os.path.join(config['dataset_path'], 'images/val')}

nc: {len(config['class_names'])}
names: {config['class_names']}
"""

with open(os.path.join(config['dataset_path'], 'data.yaml'), 'w') as f:
    f.write(data_yaml)