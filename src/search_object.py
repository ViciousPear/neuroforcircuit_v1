from ultralytics import YOLO
import cv2
from . import search_text, qf
from collections import Counter
from db_api import db_client_api
#from database import searching_in_base

db_client = db_client_api.DBClient()


colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (192, 192, 192), (128, 128, 128), (128, 128, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128),
    (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209),
    (148, 0, 211), (255, 20, 147)
]

def draw_rect(results, model, image):
    thickness = 2
    font_scale = 0.8
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    
    # Локальные списки для хранения данных
    list_for_searching = []
    list_for_quantity = []
    
    for result in results:
        boxes = []
        confidences = []
        class_ids = []

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf >= SCORE_THRESHOLD:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(conf)
                class_ids.append(int(box.cls[0]))

        if len(boxes) == 0:
            return list_for_searching, list_for_quantity

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                cls = class_ids[i]
                color = [int(c) for c in colors[cls]]

                if cls == 8:  
                    text = search_text.recognize_text_from_bbox(image, x, y, x + w, y + h)
                    if (search_text.search_qf(text) is not None):
                        qf_1 = search_text.search_qf(text)
                        list_for_searching.append(qf_1)
                    if(search_text.search_ta(text) is not None):
                        ta_1 = search_text.search_ta(text)
                        list_for_quantity.append(ta_1)

                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                text = f"{model.names[cls]} {confidences[i]:.2f}"
                (text_width, text_height) = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness
                )[0]

                text_offset_x, text_offset_y = x, y - 5
                box_coords = (
                    (text_offset_x, text_offset_y - text_height - 2),
                    (text_offset_x + text_width + 2, text_offset_y),
                )

                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
                cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

                cv2.putText(
                    image,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    thickness=thickness,
                )
    
    return list_for_searching, list_for_quantity

def detect_one_image(image_path, model):
    # Очищаем изображение перед обработкой
    image = cv2.imread(image_path)
    
    # Создаем локальные списки для каждого вызова
    request_results = []
    request_wh_results = []
    request_trans_results = []
    results_of_searching = []
    results_list = []
    
    results = model(image, conf=0.5, classes=[0,1,2,8])
    
    # Получаем списки из draw_rect
    list_for_searching, list_for_quantity = draw_rect(results, model, image)
    
    detected_classes = results[0].boxes.cls.tolist()
    class_counts = Counter(detected_classes)
    
    quantity_qf = class_counts.get(0.0, 0)
    if quantity_qf > len(list_for_searching):
        difference = quantity_qf - len(list_for_searching)
        for i in range(difference):
            list_for_searching.append(qf.create_qf('', '', '', ''))

    for qf_obj in list_for_searching:
        print(f"ID: {repr(qf_obj.ID_QF)}, \
             Ток: {repr(qf_obj.Current)},\
             Напряжение: {repr(qf_obj.Voltage)},\
             Откл. способность: {repr(qf_obj.Current_Close)}")
        request_results.append(db_client.search_breakers(qf_obj))

    if (class_counts.get(2.0, 0) != 0):
        for i in range(class_counts.get(2.0, 0)):
            request_wh_results.append(db_client.search_counters())

    if (class_counts.get(1.0, 0) != 0) or len(list_for_quantity) != 0:
        quantity_trans = class_counts.get(1.0, 0)
        if len(list_for_quantity) != 0:
            for ta_obj in list_for_quantity:
                quantity_trans += ta_obj.ta_quantity

        for i in range(quantity_trans):  
            request_trans_results.append(db_client.search_transformators())

    # Вывод результатов (опционально)
    if (len(request_results) != 0):
        print('Найденные автоматические выключатели: ')
        for index, elements in enumerate(request_results, 1):
            print('Позиция', index)
            print(elements)
            print('\n')

    if (len(request_trans_results) != 0):
        print('Найденные трансформаторы тока: ')
        for index, elements_trans in enumerate(request_trans_results, 1):
            print('Позиция', index)
            for elem in elements_trans:
                print(elem)
            print('\n')

    if (len(request_wh_results) != 0):
        print('Найденные счетчики тока: ')
        for index, elements_wh in enumerate(request_wh_results, 1):
            print('Позиция', index)
            for elem in elements_wh:
                print(elem)
            print('\n')
    
    # Формируем результаты
    results_of_searching.append(request_results)
    results_of_searching.append(request_trans_results)
    results_of_searching.append(request_wh_results)
    results_list.append(image)
    results_list.append(results_of_searching)
    
    return results_list



# if __name__ == "__main__":
#     model = YOLO("./runs/restudying_neuro_v5.71s/weights/best.pt") 
#     result = detect_one_image("для тестов/0102.png", model)