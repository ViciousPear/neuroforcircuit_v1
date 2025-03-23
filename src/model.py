from ultralytics import YOLO
from PIL import Image
import numpy
import cv2
import os
import torch
import os
import psutil
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 128, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

list_for_searching = []

class QF():
    __ID_QF = ''
    __Current = ''
    __Voltage = ''
    __Current_Close = ''

    def __init__(self, ID_QF, Current, Voltage, Current_Close):
        self.__ID_QF = ID_QF
        self.__Current = Current
        self.__Voltage = Voltage
        self.__Current_Close = Current_Close

    @property
    def ID_QF(self):
        return self.__ID_QF

    @property 
    def Current(self):
        return self.__Current
    
    @property
    def Voltage(self):
        return self.__Voltage
    
    @property
    def Current_Close(self):
        return self.__Current_Close
    
    @ID_QF.setter
    def ID_QF(self, ID_QF):
        self.__ID_QF = ID_QF

    @Current.setter
    def Current(self, Current):
        self.__Current = Current

    @Voltage.setter
    def Voltage(self, Voltage):
        self.__Voltage = Voltage

    @Current_Close.setter
    def Current_Close(self, Current_Close):
        self.__Current_Close = Current_Close    
  
    def print_data(self):
        print(self.__ID_QF, ':', self.__Current, self.__Voltage)


class trans_TA:
    __ta_name = ''
    __ta_quantity = ''    

def create_qf(ID_QF, Current, Voltage, Current_Close):
    new_qf = QF(ID_QF, Current, Voltage, Current_Close)  # Создаем новый объект QF
    return new_qf


def learning_neuro():
    print("Проверка CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Название GPU:", torch.cuda.get_device_name(0))
    
    # Явно указываем индекс GPU
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO('./runs/restudying_neuro_v5.61s/weights/best.pt')
    model.train(
        data='data.yaml',
        epochs=20,
        imgsz=1440,
        name='restudying_neuro_v5.61s',
        patience=7,
        batch=10,  # Уменьшенный размер батча
        device='cuda',  # Теперь передаётся как 0 или 'cpu'
        project='./runs',
        workers=8
        #amp=False  # Временно отключено для теста
    )
    
def recognize_text_from_bbox(image, x1, y1, x2, y2):
    #получение изображений по координатам
    roi = image[y1:y2, x1:x2] 

    # Увеличение на 200%
    scale_percent = 200  
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(roi, (width, height), interpolation=cv2.INTER_CUBIC)
   
    # Преобразование в grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
   
    # Бинарирование
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Создание структурного элемента (ядро для морфологических операций)
    kernel = numpy.ones((3, 3), numpy.uint8)

    # Морфологическое расширение (dilation)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Морфологическое сужение (erosion)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    text = pytesseract.image_to_string(eroded_image, config='--oem 3 -c tessedit_char_whitelist="ABCFGQTmok0123456789/-зк')  # Запуск OCR

    return text.strip()
    

def search_qf(text):
    id_qf = r"\b\d*[OQ0]F[TG]?\d*\.?\d*\b"  # Идентификатор (например, QF19)
    current_pattern = r"(\d+\s*-\s*\d+\s*A|\d+\s*A)"
    voltage_pattern = r"(AC\d+/\d{3}|AC\w+)"
    current_close_pattern = r"\d{3}(?=\D*kA)"
    device_id = ''.join(re.findall(id_qf, text))
    current_range = ''.join(re.findall(current_pattern, text)) 
    current_voltage = ''.join(re.findall(voltage_pattern, text)) + 'В'\
    if re.findall(voltage_pattern, text) else ''
    current_close = ''.join(re.findall(current_close_pattern, text))
    if (device_id != '' or current_range != '' or current_voltage != '' or current_close != ''):
        new_qf = create_qf(device_id, current_range, current_voltage, current_close)
        return new_qf
    return
    

    
def draw_rect(results, model, image):
    thickness = 2
    font_scale = 0.8
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
   
    for result in results:
        boxes = []
        confidences = []
        class_ids = []  # Храним классы для дальнейшего использования

        # Сбор всех боксов и уверенности
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf >= SCORE_THRESHOLD:  # Отбрасываем слишком неуверенные боксы
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # (x, y, width, height)
                confidences.append(conf)
                class_ids.append(int(box.cls[0]))  # Сохраняем класс

        # Проверяем, есть ли боксы вообще
        if len(boxes) == 0:
            return

        # Фильтрация NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        if len(idxs) > 0:
            for i in idxs.flatten():  # Идём по индексам, оставшимся после NMS
                x, y, w, h = boxes[i]  # Координаты бокса
                cls = class_ids[i]  # Класс объекта
               
                color = [int(c) for c in colors[cls]]  # Цвет для класса

                if cls == 8:  # Если класс объекта соответствует нужному (например, класс 8)
                    # Вызовем функцию для распознавания текста в рамках bounding box
                    text = recognize_text_from_bbox(image, x, y, x + w, y + h)
                    print(f"Объект {model.names[cls]} ({conf:.2f}): {text}")
                    if (search_qf(text) != None):
                        qf = search_qf(text)
                        # text_split = text.split('\n\n')
                        # text_split = list(filter(lambda x: x != '', text_split))
                        list_for_searching.append(qf)

                # Отрисовка bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

                # Подготовка текста
                text = f"{model.names[cls]} {confidences[i]:.2f}"
                (text_width, text_height) = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness
                )[0]

                # Координаты фона для текста
                text_offset_x, text_offset_y = x, y - 5
                box_coords = (
                    (text_offset_x, text_offset_y - text_height - 2),
                    (text_offset_x + text_width + 2, text_offset_y),
                )

                # Добавление фона (НЕ перезаписываем image!)
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
                cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)  # Модифицируем image напрямую

                # Отображение текста
                cv2.putText(
                    image,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    thickness=thickness,
                )


def detect_one_image(image_path, model):

    # Загрузить изображение
    image = cv2.imread(image_path)
    # Запустить модель YOLO на изображении
   
    results = model(image, conf=0.5, classes=[0,1,2,8])

    draw_rect(results, model, image)

    for qf in list_for_searching:
        print(f"ID: {qf.ID_QF}, Ток: {qf.Current}, Напряжение: {qf.Voltage}, З. ток: {qf.Current_Close}")

     # Развернуть окно на весь экран
    window_name = "YOLOv8 Image Detection"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Показать изображение
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(path, test_image):

 # Предобученная модель
    model = YOLO('./runs/restudying_neuro_v5.61s/weights/best.pt')
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



if __name__ == '__main__':
    learning_neuro()
    # model = YOLO("./runs/restudying_neuro_v5.61s/weights/best.pt") 
    # detect_one_image("tests/РУ_0,4_кВ_3200А_в_комплекте_с_шинным_мостами_ТП_49_ЭЛ_ЩИТ_17_06 (6)-05.png", model)
    
    # folder_path = "./tests"
    # img_list = []

    # for images in os.listdir(folder_path):
    #     if(images.endswith('.png')):
    #         img_list.append(images)
    # folder_path += '/'
    # print(img_list)
    # for i in range(0, len(img_list)):
    #     process_image(folder_path, img_list[i])

    #print("Физические ядра:", psutil.cpu_count(logical=False))
    #print("Логические ядра:", psutil.cpu_count(logical=True))
