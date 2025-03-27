from ultralytics import YOLO
import numpy
import cv2
import torch
import os


colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 128, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]



def learning_neuro():
    print("Проверка CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Название GPU:", torch.cuda.get_device_name(0))
    
    # Явно указываем индекс GPU
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO('./runs/restudying_neuro_v5.61s/weights/best.pt')
    model.train(
        data='data.yaml',
        epochs=25,
        imgsz=1440,
        name='restudying_neuro_v5.71s',
        patience=7,
        batch=10,  # Уменьшенный размер батча
        device='cuda',  # Теперь передаётся как 0 или 'cpu'
        project='./runs',
        workers=8
        #amp=False  # Временно отключено для теста
    )


def process_image(path, test_image):

 # Предобученная модель
    model = YOLO('./runs/restudying_neuro_v5.71s/weights/best.pt')
     # Загрузка изображения
    image = cv2.imread(os.path.join(path, test_image))
    original_height, original_width = image.shape[:2]  # Сохраняем исходный размер изображения
    thickness = 1
    font_scale = 0.5
    confidences = 0.55
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.5
    # Применение модели
    results = model(image, conf=0.01,classes=[0,1,2,8])[0]

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

        for result in results:
         boxes = []
         confidences = []
         class_ids = []  # Храним классы для дальнейшего использования

        # Сбор боксов и уверенности
         for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # Формат (x, y, width, height)
            confidences.append(conf)
            class_ids.append(int(box.cls[0]))

        # Фильтрация NMS
         idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

         if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]  # Получаем координаты бокса
                cls = int(result.boxes[i].cls[0])  # Класс объекта

                color = [int(c) for c in colors[cls]]  # Цвет для класса

                # Нарисовать bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

                # Подготовка текста
                text = f"{model.names[cls]} {confidences[i]:.2f}"
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                            fontScale=font_scale, thickness=thickness)[0]

                # Координаты фона для текста
                text_offset_x, text_offset_y = x, y - 5
                box_coords = ((text_offset_x, text_offset_y - text_height - 2), (text_offset_x + text_width + 2, text_offset_y))

                # Добавление полупрозрачного фона
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

                # Отображение текста
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=font_scale, color=(255, 255, 255), thickness=thickness)

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



    learning_neuro()
    
    # folder_path = "./tests"
    # img_list = []

    # for images in os.listdir(folder_path):
    #     if(images.endswith('.png')):
    #         img_list.append(images)
    # folder_path += '/'
    # print(img_list)
    # for i in range(0, len(img_list)):
    #     process_image(folder_path, img_list[i])

    # print("Физические ядра:", psutil.cpu_count(logical=False))
    # print("Логические ядра:", psutil.cpu_count(logical=True))
