import os
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# ========================
# Конфигурация и утилиты
# ========================
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
IGNORE_EXTENSIONS = ('.npy', )

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)

def should_ignore(filename: str) -> bool:
    return filename.lower().endswith(IGNORE_EXTENSIONS)

# ========================
# Основные функции
# ========================

def create_yolo_dataset(
    source_images_dir,
    source_labels_dir,
    output_dir="data",
    train_ratio=0.8,
    classes=None,
    seed=42
):
    """Создает структурированный YOLO датасет"""
    # Создание директорий
    os.makedirs(f"{output_dir}/train/images", exist_ok=True)
    os.makedirs(f"{output_dir}/train/labels", exist_ok=True)
    os.makedirs(f"{output_dir}/val/images", exist_ok=True)
    os.makedirs(f"{output_dir}/val/labels", exist_ok=True)
  

    # Сбор данных с фильтрацией
    valid_pairs = []
    for img_file in os.listdir(source_images_dir):
        if should_ignore(img_file) or not is_image_file(img_file):
            continue
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(source_labels_dir, label_file)
        
        if os.path.exists(label_path):
            valid_pairs.append((img_file, label_file))

    # Разделение данных
    random.seed(seed)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    # Копирование файлов
    def copy_files(pairs, dataset_type):
        for img_file, label_file in pairs:
            shutil.copy(
                os.path.join(source_images_dir, img_file),
                os.path.join(output_dir, dataset_type, "images", img_file)
            )
            shutil.copy(
                os.path.join(source_labels_dir, label_file),
                os.path.join(output_dir, dataset_type, "labels", label_file)
            )

    copy_files(train_pairs, "train")
    copy_files(val_pairs, "val")

    # Генерация data.yaml
    classes = classes or sorted({
        int(line.split()[0])
        for label_file in os.listdir(source_labels_dir)
        for line in open(os.path.join(source_labels_dir, label_file))
    })
    
    yaml_content = f"""names: {classes}
nc: {len(classes)}
train: {os.path.abspath(output_dir)}/train/images
val: {os.path.abspath(output_dir)}/val/images
"""
    with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
        f.write(yaml_content)

def augment_dataset(
    source_images_dir,
    source_labels_dir,
    output_dir,
    multiply_factor=4,
    augmentations=None
):
    """Генерация аугментированных данных"""
    # Создаем директории для аугментированных данных
    aug_images_dir = os.path.join(output_dir, "aug_images")
    aug_labels_dir = os.path.join(output_dir, "aug_labels")
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)

    # Исправленные аугментации (убрана CoarseDropout)
    if augmentations is None:
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomShadow(p=0.1),
            A.ToGray(p=0.1),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Обработка изображений
    image_files = [f for f in os.listdir(source_images_dir) 
                 if is_image_file(f) and not should_ignore(f)]
    
    for img_name in tqdm(image_files, desc="Augmenting"):
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(source_labels_dir, f"{base_name}.txt")
        
        try:
            # Чтение данных с проверкой существования файлов
            if not os.path.exists(label_path):
                print(f"Warning: Missing label for {img_name}")
                continue

            image = cv2.imread(os.path.join(source_images_dir, img_name))
            if image is None:
                print(f"Error: Failed to read image {img_name}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes = []
            class_labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Invalid label format in {label_path}")
                        continue
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:5]))
                    bboxes.append(bbox)
                    class_labels.append(class_id)

            # Генерация аугментаций
            for i in range(multiply_factor - 1):
                try:
                    augmented = augmentations(
                        image=image, 
                        bboxes=bboxes, 
                        class_labels=class_labels
                    )
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']

                    # Генерация уникального имени файла
                    aug_img_name = f"{base_name}_aug{i}_{random.randint(0, 1000)}.jpg"
                    aug_img_path = os.path.join(aug_images_dir, aug_img_name)
                    cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    
                    # Сохранение аннотаций
                    aug_label_path = os.path.join(aug_labels_dir, f"{os.path.splitext(aug_img_name)[0]}.txt")
                    with open(aug_label_path, 'w') as f:
                        for bbox, cls_id in zip(aug_bboxes, class_labels):
                            f.write(f"{cls_id} {' '.join(map(str, bbox))}\n")
                except Exception as e:
                    print(f"Error processing {img_name} (aug{i}): {str(e)}")

        except Exception as e:
            print(f"Critical error processing {img_name}: {str(e)}")
            continue

def create_extended_dataset(
    source_images_dir,
    source_labels_dir,
    output_dir="data_augmented",
    train_ratio=0.8,
    multiply_factor=4,
    seed=42
):
    """Создает расширенный датасет с аугментацией"""
    # Настройка путей
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Генерация аугментированных данных
    augment_dataset(
        source_images_dir=source_images_dir,
        source_labels_dir=source_labels_dir,
        output_dir=temp_dir,
        multiply_factor=multiply_factor
    )

    # Объединение данных
    combined_images = os.path.join(temp_dir, "combined_images")
    combined_labels = os.path.join(temp_dir, "combined_labels")
    os.makedirs(combined_images, exist_ok=True)
    os.makedirs(combined_labels, exist_ok=True)

    # Копирование оригинальных данных
    for f in os.listdir(source_images_dir):
        if not should_ignore(f):
            shutil.copy(
                os.path.join(source_images_dir, f),
                combined_images
            )
    for f in os.listdir(source_labels_dir):
        if not should_ignore(f):
            shutil.copy(
                os.path.join(source_labels_dir, f),
                combined_labels
            )

    # Копирование аугментированных данных
    aug_images = os.path.join(temp_dir, "aug_images")
    aug_labels = os.path.join(temp_dir, "aug_labels")
    for f in os.listdir(aug_images):
        shutil.copy(
            os.path.join(aug_images, f),
            combined_images
        )
    for f in os.listdir(aug_labels):
        shutil.copy(
            os.path.join(aug_labels, f),
            combined_labels
        )

    # Создание финального датасета
    create_yolo_dataset(
        source_images_dir=combined_images,
        source_labels_dir=combined_labels,
        output_dir=output_dir,
        train_ratio=train_ratio,
        seed=seed
    )

    # Очистка временных файлов
    shutil.rmtree(temp_dir)

# ========================
# Пример использования
# ========================
if __name__ == "__main__":
    create_extended_dataset(
        source_images_dir="data/train/images",
        source_labels_dir="data/train/labels",
        output_dir="my_dataset_x4",
        multiply_factor=4,
        train_ratio=0.8,
        seed=42
    )