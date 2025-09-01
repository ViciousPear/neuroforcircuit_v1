import requests
import cv2
from collections import Counter
import db_client_api
import os
import qf, ta
import search_text

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
    
    # Собираем детекции
    text_boxes = []
    automatics_boxes = []
    other_boxes = []
    
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
                
            box_data = (x1, y1, x2, y2, conf, cls_id, (x1+x2)/2, (y1+y2)/2)
            
            if cls_id == 8:  # text
                text_boxes.append(box_data)
            elif cls_id == 0:  # automatics
                automatics_boxes.append(box_data)
            else:
                other_boxes.append(box_data)
    
    # Сортируем все элементы по вертикали (Y), затем по горизонтали (X)
    text_boxes.sort(key=lambda b: (b[7], b[6]))
    automatics_boxes.sort(key=lambda b: (b[7], b[6]))
    other_boxes.sort(key=lambda b: (b[7], b[6]))

    # 1. Связываем automatics с текстами (справа)
    auto_text_pairs = []
    used_texts = set()
    
    for auto in automatics_boxes:
        ax1, ay1, ax2, ay2, aconf, acls, acx, acy = auto
        
        best_text = None
        min_distance = float('inf')
        
        for text in text_boxes:
            if text in used_texts:
                continue
                
            tx1, ty1, tx2, ty2, tconf, tcls, tcx, tcy = text
            
            # Текст должен быть справа и в пределах вертикального overlap
            if (tcx > acx) and (ty1 < ay2) and (ty2 > ay1):
                distance = abs(tcx - acx)
                if distance < min_distance:
                    min_distance = distance
                    best_text = text
        
        if best_text and min_distance < 300:  # Макс расстояние 300px
            auto_text_pairs.append((auto, best_text))
            used_texts.add(best_text)
    
    # 2. Назначаем номера текстам (по порядку расположения)
    text_numbers = {}
    for idx, text in enumerate(text_boxes, 1):
        text_numbers[text] = idx
    
    # 3. Назначаем номера automatics (по порядку расположения, но с привязкой к текстам)
    box_numbers = {}
    auto_counter = 1
    
    # Сначала automatics с привязанными текстами
    for auto, text in sorted(auto_text_pairs, key=lambda pair: (pair[0][7], pair[0][6])):
        box_numbers[auto] = text_numbers[text]
    
    # Затем оставшиеся automatics
    for auto in automatics_boxes:
        if auto not in box_numbers:
            box_numbers[auto] = auto_counter
            auto_counter += 1
    
    # 4. Нумеруем другие элементы (после automatics)
    other_counter = auto_counter
    for other in other_boxes:
        box_numbers[other] = other_counter
        other_counter += 1
    
    local_text_url = "http://localhost:8000"
    text_service_docker = "http://text-service:5002"
    text_service_url = os.getenv("TEXT_SERVICE_URL", text_service_docker)
    
    # Подготавливаем текстовые чанки для отправки
    text_chunks = []
    
    # Отрисовка и обработка текста
    for text in text_boxes:
        x1, y1, x2, y2, conf, cls_id, cx, cy = text
        color = colors[cls_id % len(colors)]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # ВЫЗОВ ВАШЕЙ ФУНКЦИИ recognize_text_from_bbox
        text_content = search_text.recognize_text_from_bbox(image, x1, y1, x2, y2)
        
        if text_content:
            # Добавляем в chunks для отправки в text-service
            text_chunks.append({
                "text": text_content,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls_id)
            })
    
    # Отправляем распознанный текст в text-service для обработки
    if text_chunks:
        try:
            response = requests.post(
                f"{text_service_url}/api/process-text-batch", 
                json={
                    "chunks": text_chunks,
                    "image_size": [image.shape[1], image.shape[0]],
                    "image_shape": list(image.shape)
                },
                timeout=60
            )
            response.raise_for_status()
            
            # Обрабатываем результаты
            batch_results = response.json()
            
            for result in batch_results:
                if result.get("type") == "circuit_breaker":
                    qf_dict = result.get("object", {})
                    qf_obj = qf.create_qf(
                        qf_dict.get("ID_QF", ""),
                        qf_dict.get("Current", ""),
                        qf_dict.get("Voltage", ""),
                        qf_dict.get("Current_Close", "")
                    )
                    list_for_searching.append(qf_obj)
                    
                elif result.get("type") == "current_transformer":
                    ta_dict = result.get("object", {})
                    # Исправляем получение имени для приватного поля
                    ta_name = ta_dict.get("_Trans_TA__ta_name", "")
                    ta_obj = ta.create_ta(ta_name)
                    list_for_quantity.append(ta_obj)
                    
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса к text-service: {e}")
    
    # Отрисовка всех элементов с номерами
    all_boxes = automatics_boxes + other_boxes
    all_boxes.sort(key=lambda b: (b[7], b[6]))  # Сортировка по Y, затем X
    
    for box in all_boxes:
        x1, y1, x2, y2, conf, cls_id, cx, cy = box
        color = colors[cls_id % len(colors)]
        label = str(box_numbers[box])
        
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
   yolo_false =  "http://82.202.129.245:5001"
   try:
        response = requests.post(
            f"{yolo_false}/detect_raw",
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
            print(qf_obj)
            print(f"ID: {repr(qf_obj.ID_QF if qf_obj else '')}, "
                f"Ток: {repr(qf_obj.Current if qf_obj else '')}, "
                f"Напряжение: {repr(qf_obj.Voltage if qf_obj else '')}, "
                f"Откл. способность: {repr(qf_obj.Current_Close if qf_obj else '')}")
            
            # Передаем словарь напрямую
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



if __name__ == "__main__":
    #model = YOLO("./runs/restudying_neuro_v5.71s/weights/best.pt") 
    result = detect_one_image("C:/nekitlox/NeuroForCircuit/tests/RU.png")