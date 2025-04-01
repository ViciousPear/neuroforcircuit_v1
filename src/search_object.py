from ultralytics import YOLO
import cv2
import search_text


list_for_searching = []
list_for_quantity = []

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 128, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

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
                    text = search_text.recognize_text_from_bbox(image, x, y, x + w, y + h)
                    # print(f"Объект {model.names[cls]} ({conf:.2f}): {text}")
                    if (search_text.search_qf(text) != None):
                        qf_1 = search_text.search_qf(text)
                        # text_split = text.split('\n\n')
                        # text_split = list(filter(lambda x: x != '', text_split))
                        list_for_searching.append(qf_1)
                    if(search_text.search_ta(text) != None):
                        ta_1 = search_text.search_ta(text)
                        list_for_quantity.append(ta_1)

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
    
    for qf_obj in list_for_searching:
        print(f"ID: {qf_obj.ID_QF}, Ток: {qf_obj.Current}, Напряжение: {qf_obj.Voltage}, Откл. способность: {qf_obj.Current_Close}")

    for ta_obj in list_for_quantity:
        print(f"ID: {ta_obj.ta_name}, Количество: {ta_obj.ta_quantity}")

     # Развернуть окно на весь экран
    window_name = "YOLOv8 Image Detection"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Показать изображение
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# if __name__ == '__main__':

#     model = YOLO("./runs/restudying_neuro_v5.71s/weights/best.pt") 
#     detect_one_image("not_based.png", model)