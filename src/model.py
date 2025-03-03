from ultralytics import YOLO
from PIL import Image
import numpy
import cv2
import os
import torch
import os
import psutil


colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

def learning_neuro():
    print("Проверка CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Название GPU:", torch.cuda.get_device_name(0))
    
    # Явно указываем индекс GPU
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO('C:/neuroforcircuit_v1/runs/restudying_neuro_v4.2s/weights/best.pt')
    model.train(
        data='data.yaml',
        epochs=150,
        imgsz=800,
        name='restudying_neuro_model_s_v4.2',
        patience=10,
        batch=16,  # Уменьшенный размер батча
        device='cuda',  # Теперь передаётся как 0 или 'cpu'
        project='./runs',
        #amp=False  # Временно отключено для теста
    )
    
def analytics_learning():
    model = YOLO("yolov8s.pt")  # Загружаем новую модель

    pretrained_weights = YOLO("./runs/circuit_elements/weights/best.pt").model.state_dict()
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_weights, strict=False)

    # Выводим несовпадающие слои
    print("Пропущенные слои (missing_keys):", len(missing_keys))
    print("Лишние слои в весах (unexpected_keys):", len(unexpected_keys))                                   

    if len(missing_keys) > len(model.model.state_dict()) * 0.5:
        print("Слишком много пропущенных слоев. Рекомендуется обучить модель с нуля.")

def process_image(path, test_image):

 # Предобученная модель
    model = YOLO('./runs/restudying_neuro_model_s_v4.24/weights/best.pt')
     # Загрузка изображения
    image = cv2.imread(os.path.join(path, test_image))
    original_height, original_width = image.shape[:2]  # Сохраняем исходный размер изображения

    # Применение модели
    results = model(image)[0]

    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(numpy.int32)

    # Масштабирование bounding boxes к исходному размеру изображения
    scale_x = original_width / results.orig_shape[1]
    scale_y = original_height / results.orig_shape[0]
    boxes = boxes * numpy.array([scale_x, scale_y, scale_x, scale_y])
    boxes = boxes.astype(numpy.int32)

    # Словарь для группировки результатов
    grouped_objects = {}

    # Рисование рамок и группировка результатов
    for class_id, box in zip(classes, boxes):
        class_name = classes_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]
        if class_name not in grouped_objects:
            grouped_objects[class_name] = []
        grouped_objects[class_name].append(box)

        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Сохранение измененного изображения
    new_image_path = os.path.join("./yolo_image+text/", os.path.splitext(test_image)[0] + '_yolo' + os.path.splitext(test_image)[1])
    cv2.imwrite(new_image_path, image)

    # Сохранение данных в текстовый файл
    text_file_path = os.path.join("./yolo_image+text/", os.path.splitext(test_image)[0] + '_yolo' + '_data.txt')
    with open(text_file_path, 'w') as f:
        for class_name, details in grouped_objects.items():
            f.write(f"{class_name}:\n")
            for detail in details:
                f.write(f"Coordinates: ({detail[0]}, {detail[1]}, {detail[2]}, {detail[3]})\n")

    print(f"Processed {test_image}:")
    print(f"Saved bounding-box image to {new_image_path}")
    print(f"Saved data to {text_file_path}")



if __name__ == '__main__':
    #learning_neuro()
   # process_image()


    folder_path = "./tests"
    img_list = []

    for images in os.listdir(folder_path):
        if(images.endswith('.png')):
            img_list.append(images)
    folder_path += '/'
    print(img_list)
    for i in range(0, len(img_list)):
        process_image(folder_path, img_list[i])

#print("Физические ядра:", psutil.cpu_count(logical=False))
#print("Логические ядра:", psutil.cpu_count(logical=True))
