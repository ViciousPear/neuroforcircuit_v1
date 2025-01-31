from ultralytics import YOLO
from PIL import Image
import numpy
import cv2
import os

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

def learning_neiro():
    model = YOLO('yolov8n.pt')
    # Запуск обучения
    model.train(
        data='data.yaml',        # Путь к файлу конфигурации данных
        epochs=50,               # Количество эпох
        imgsz=640,               # Размер изображения (640x640)
        name='circuit_elements', # Название модели
        device='cpu'                 # Использование GPU (укажите 'cpu', если нет GPU)
    )

def process_image(test_image):

 # Предобученная модель
    model = YOLO('C:/nekitlox/NeuroForCircuit/runs/detect/circuit_elements/weights/best.pt') 
 # Загрузка изображения
    image = cv2.imread(test_image)
    # Применение модели
    results = model(image)[0]

    # Загрузка изображения
    image = cv2.imread(test_image)
    results = model(image)[0]
    
    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(numpy.int32)

    #словарь для группировки результатов
    grouped_objects = {}

    #рисование рамок и группировка результатов
    for class_id, box in zip(classes, boxes):
        class_name = classes_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]
        if class_name not in grouped_objects:
          grouped_objects[class_name] = []  
        grouped_objects[class_name].append(box)

        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохранение измененного изображения
    new_image_path = os.path.splitext(test_image)[0] + '_yolo' + os.path.splitext(test_image)[1]
    cv2.imwrite(new_image_path, image)

    # Сохранение данных в текстовый файл
    text_file_path = os.path.splitext(test_image)[0] + '_data.txt'
    with open(text_file_path, 'w') as f:
        for class_name, details in grouped_objects.items():
            f.write(f"{class_name}:\n")
            for detail in details:
                f.write(f"Coordinates: ({detail[0]}, {detail[1]}, {detail[2]}, {detail[3]})\n")

    print(f"Processed {test_image}:")
    print(f"Saved bounding-box image to {new_image_path}")
    print(f"Saved data to {text_file_path}")

process_image('test_image.png')