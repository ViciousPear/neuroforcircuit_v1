import requests
import cv2
import search_text, qf
from collections import Counter
import db_client_api
import os
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
def draw_rect(results, image):
    thickness = 2
    font_scale = 0.8
    SCORE_THRESHOLD = 0.5
    
    list_for_searching = []
    list_for_quantity = []
    
    # Собираем все детекции
    all_boxes = []
    text_boxes = []
    
    for result in results:
        if isinstance(result, dict):
            boxes_data = result['results']
            names = result['names']
        else:
            boxes_data = result.boxes.data.cpu().numpy()
            names = result.names
        
        for box in boxes_data:
            x1, y1, x2, y2, conf, cls_id = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf = float(conf)
            cls_id = int(cls_id)
            
            if conf < SCORE_THRESHOLD:
                continue
                
            if cls_id == 8:  # text
                text_boxes.append((x1, y1, x2, y2, conf, cls_id))
            else:
                all_boxes.append((x1, y1, x2, y2, conf, cls_id))
    
    # Сортируем текстовые блоки по вертикали (сверху вниз)
    text_boxes.sort(key=lambda b: b[1])
    
    # Создаем список для нумерации (класс: порядковый номер)
    class_numbers = {}
    current_number = 1
    
    # Сначала обрабатываем текстовые блоки и связанные с ними объекты
    for text_box in text_boxes:
        tx1, ty1, tx2, ty2, tconf, tcls = text_box
        text_center_y = (ty1 + ty2) / 2
        
        # Находим ближайший объект (не text) для этого текста
        closest_obj = None
        min_distance = float('inf')
        
        for box in all_boxes:
            x1, y1, x2, y2, conf, cls_id = box
            if cls_id == 8:  # Пропускаем text
                continue
                
            box_center_y = (y1 + y2) / 2
            distance = abs(box_center_y - text_center_y)
            
            if distance < min_distance:
                min_distance = distance
                closest_obj = box
        
        # Если нашли связанный объект, присваиваем номер
        if closest_obj:
            x1, y1, x2, y2, conf, cls_id = closest_obj
            if (x1, y1, x2, y2, cls_id) not in class_numbers:
                class_numbers[(x1, y1, x2, y2, cls_id)] = current_number
                current_number += 1
    
    # Затем нумеруем оставшиеся объекты (не связанные с текстом)
    for box in all_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        if cls_id == 8 or (x1, y1, x2, y2, cls_id) in class_numbers:
            continue
        class_numbers[(x1, y1, x2, y2, cls_id)] = current_number
        current_number += 1
    
    # Отрисовка всех объектов
    for box in all_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        color = colors[cls_id % len(colors)]
        
        if (x1, y1, x2, y2, cls_id) in class_numbers:
            label = f"{class_numbers[(x1, y1, x2, y2, cls_id)]}"
        else:
            label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    # Отрисовка текстовых блоков (без номеров)
    for box in text_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        color = colors[cls_id % len(colors)]
        
        # Распознаем текст
        text = search_text.recognize_text_from_bbox(image, x1, y1, x2, y2)
        if text:
            qf = search_text.search_qf(text)
            if qf is not None:
                list_for_searching.append(qf)
            ta = search_text.search_ta(text)
            if ta is not None:
                list_for_quantity.append(ta)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    return list_for_searching, list_for_quantity


def detect_one_image(image_path):
   # Создаем локальные списки для каждого вызова
   request_results = []
   request_wh_results = []
   request_trans_results = []
   results_of_searching = []
   results_list = []
   image = cv2.imread(image_path)
    
    # Отправка в YOLO-сервис
   _, img_encoded = cv2.imencode('.png', image)
   files = {'file': ('image.png', img_encoded.tobytes(), 'image/png')}
   yolo_url = os.getenv("YOLO_API_URL", "http://yolo-service:5001")
   try:
        response = requests.post(
            f"{yolo_url}/detect_raw",
            files=files,
            timeout=600
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get('success'):
            raise ValueError("YOLO service returned error")
        
        results = [{
            'results': data['results'],
            'names': data['names']
        }]
        
        list_for_searching, list_for_quantity = draw_rect(results, image)
        
        # Подсчет классов
        class_ids = [int(box[5]) for box in data['results']]
        class_counts = Counter(class_ids)
        
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
        
        # print(request_results)
        # print(request_trans_results)
        # print(request_wh_results)

        # Формируем результаты
        results_of_searching.append(request_results)
        results_of_searching.append(request_trans_results)
        results_of_searching.append(request_wh_results)
        results_list.append(image)
        results_list.append(results_of_searching)
        
        return results_list

   except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise



# if __name__ == "__main__":
#     model = YOLO("./runs/restudying_neuro_v5.71s/weights/best.pt") 
#     result = detect_one_image("для тестов/0102_ПР_ШУ,_КТП_АБК№1_после_замен_11_17_1-1.png", model)