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
    """
    Запускает дообучение (fine-tuning) YOLO-модели на датасете data.yaml.

    Логика:
        - Проверяет доступность CUDA и печатает имя GPU, если он доступен.
        - Загружает предобученные веса из
          './runs/restudying_neuro_v5.61s/weights/best.pt'.
        - Запускает обучение с фиксированными гиперпараметрами:
            epochs=25, imgsz=1440, batch=10, patience=7,
            device='cuda', workers=8, project='./runs',
            name='restudying_neuro_v5.71s'.

    Args:
        Нет.

    Returns:
        None: Результаты обучения сохраняются в директорию ./runs.

    Notes:
        - Предполагается, что файл data.yaml лежит в текущей рабочей
          директории.
        - device='cuda' может не сработать на машине без GPU; для CPU
          стоит заменить на device='cpu'.
        - В исходном ноутбуке была закомментирована строка amp=False —
          при проблемах с mixed precision ее можно вернуть.
    """
    print("Проверка CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Название GPU:", torch.cuda.get_device_name(0))

    # Явно указываем индекс GPU
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLO('./runs/restudying_neuro_v5.61s/weights/best.pt')
    model.train(
        data='data.yaml',
        epochs=25,
        imgsz=1440,
        name='...', # здесь надо указать название
        patience=7,
        batch=12,  
        device='cuda',  
        project='./runs',
        workers=8
    )


def process_image(path, test_image):
    """
    Прогоняет одно изображение через YOLO-модель, рисует bbox и сохраняет
    результаты (изображение + текстовый файл с координатами).

    ВНИМАНИЕ: функция находится в черновом состоянии (скопирована из
    ноутбука). В текущем виде она содержит логические ошибки:
        - модель загружается из заглушки '...' (нужно указать реальный путь);
        - внутри цикла по class_id, box создается вложенный цикл по results,
          который перезаписывает boxes/confidences/class_ids и в итоге
          NMS применяется к последнему результату;
        - часть кода (NMS, отрисовка через cv2.dnn.NMSBoxes) не согласована
          с верхнеуровневой отрисовкой bbox.
    Ниже описано задуманное поведение и текущие аргументы.

    Задуманный пайплайн:
        1. Загрузить изображение по пути os.path.join(path, test_image).
        2. Сохранить исходные width/height.
        3. Прогнать YOLO-модель с conf=0.01 и classes=[0,1,2,8].
        4. Получить bbox в формате xyxy и классы.
        5. Масштабировать bbox к исходному размеру изображения.
        6. Сгруппировать объекты по именам классов в grouped_objects.
        7. Применить NMS (cv2.dnn.NMSBoxes) с SCORE_THRESHOLD=0.5,
           IOU_THRESHOLD=0.5.
        8. Нарисовать bbox и подписи (class_name + confidence) на изображении.
        9. Сохранить изображение в ./yolo_image+text/<name>_yolo.<ext>.
       10. Сохранить координаты в ./yolo_image+text/<name>_yolo_data.txt.

    Args:
        path (str): Директория, в которой лежит изображение.
        test_image (str): Имя файла изображения.

    Returns:
        None: Результаты сохраняются на диск, в консоль печатается
            информация о путях сохранения.

    Notes:
        - thickness=1, font_scale=0.5, confidences=0.55 (не используется),
          IOU_THRESHOLD=0.5, SCORE_THRESHOLD=0.5.
        - Для корректной работы нужно заменить YOLO('...') на реальный
          путь к весам и привести вложенные циклы к единой логике.
    """
    # Предобученная модель
    model = YOLO('...') # здесь лучше указать версию модели

    # Загрузка изображения
    image = cv2.imread(os.path.join(path, test_image))
    original_height, original_width = image.shape[:2]  # Сохраняем исходный размер изображения
    thickness = 1
    font_scale = 0.5
    confidences = 0.55
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.5

    # Применение модели
    results = model(image, conf=0.01, classes=[0, 1, 2, 8])[0]

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
                    (text_width, text_height) = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, thickness=thickness
                    )[0]

                    # Координаты фона для текста
                    text_offset_x, text_offset_y = x, y - 5
                    box_coords = (
                        (text_offset_x, text_offset_y - text_height - 2),
                        (text_offset_x + text_width + 2, text_offset_y)
                    )

                    # Добавление полупрозрачного фона
                    overlay = image.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

                    # Отображение текста
                    cv2.putText(
                        image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(255, 255, 255), thickness=thickness
                    )

    # Сохранение измененного изображения
    new_image_path = os.path.join(
        "./yolo_image+text/",
        os.path.splitext(test_image)[0] + '_yolo' + os.path.splitext(test_image)[1]
    )
    cv2.imwrite(new_image_path, image)

    # Сохранение данных в текстовый файл
    text_file_path = os.path.join(
        "./yolo_image+text/",
        os.path.splitext(test_image)[0] + '_yolo' + '_data.txt'
    )
    with open(text_file_path, 'w') as f:
        for class_name, details in grouped_objects.items():
            f.write(f"{class_name}:\n")
            for detail in details:
                f.write(f"Coordinates: ({detail[0]}, {detail[1]}, {detail[2]}, {detail[3]})\n")

    print(f"Processed {test_image}:")
    print(f"Saved bounding-box image to {new_image_path}")
    print(f"Saved data to {text_file_path}")


    # learning_neuro()

    # folder_path = "./tests"
    # img_list = []

    # for images in os.listdir(folder_path):
    #     if images.endswith('.png'):
    #         img_list.append(images)
    # folder_path += '/'
    # print(img_list)
    # for i in range(0, len(img_list)):
    #     process_image(folder_path, img_list[i])

    # print("Физические ядра:", psutil.cpu_count(logical=False))
    # print("Логические ядра:", psutil.cpu_count(logical=True))