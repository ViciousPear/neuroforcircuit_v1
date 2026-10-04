import cv2
from collections import Counter
import os
import qf
import circuit_graph as cg
from services_client import send_to_yolo_service, recognize_with_text_service
from visualization import render_circuit_visualization
from geometry import determine_mounting_type_and_bbox
from graph_mapper import create_graphs, create_a_mind_map, create_connect_text_chunks


class YOLOClassID:
    """
    Сопоставление имен классов YOLO-модели с их числовыми идентификаторами.

    Используется для маршрутизации bounding box по типам при разборе
    результатов детекции (см. results_boxes).

    Attributes:
        AUTOMATIC (int): ID класса автоматических выключателей (0).
        TRANSFORMATOR (int): ID класса трансформаторов (1).
        WH_COUNTER (int): ID класса счетчиков (2).
        BRACING (int): ID класса bracing-элементов (4).
        TEXT (int): ID класса текстовых блоков (7).
        RCD (int): ID класса УЗО (9).
    """
    AUTOMATIC = 0
    TRANSFORMATOR = 1
    WH_COUNTER = 2
    BRACING = 4
    TEXT = 7
    RCD = 9


def recognize_only(image_path):
    """
    Выполняет распознавание схемы по изображению БЕЗ поиска в базе данных.

    Полный пайплайн:
        1. Читает изображение через OpenCV.
        2. Кодирует его в PNG и отправляет в YOLO-сервис
           (URL из переменной окружения YOLO_API_URL).
        3. Передает результаты детекции в analyze_circuit_diagram, который
           строит граф схемы, определяет тип монтажа, нумерует элементы
           и получает распознанные тексты через text-service.
        4. Подсчитывает количество автоматов по детекции (class_id = 0).
        5. Дополняет список QF пустыми объектами, если текстов меньше,
           чем найденных автоматов.
        6. Формирует единый список detected_elements с элементами типов
           "QF", "transformer", "counter" в формате, пригодном для API.

    Args:
        image_path (str): Путь к изображению схемы.

    Returns:
        tuple[np.ndarray, list[dict]]:
            - image: исходное изображение (OpenCV BGR).
            - detected_elements: список распознанных элементов, где каждый
              элемент — словарь вида:
                {
                    "id": str,
                    "type": "QF" | "transformer" | "counter",
                    "number": int,
                    "confidence": float,
                    "parameters": dict
                }

    Raises:
        Exception: Пробрасывает любое исключение, возникшее при обработке
            изображения (после логирования и traceback).
    """
    results_of_searching = []
    results_list = []
    image = cv2.imread(image_path)

    # Отправка в YOLO-сервис
    _, img_encoded = cv2.imencode('.png', image)
    yolo_url = os.getenv("YOLO_API_URL", "http://yolo-service:5001")
    # yolo_false = "http://номер_хоста:5001"
    # yolo_local = "http://localhost:5001" - для проверки

    try:
        # Отправка в YOLO-сервис
        yolo_data = send_to_yolo_service(image, yolo_url)

        results = [{
            'results': yolo_data['results'],
            'names': yolo_data['names']
        }]

        list_for_searching, list_for_quantity, mounting_types = analyze_circuit_diagram(results, image)

        # Подсчет классов
        class_ids = [int(box[5]) for box in yolo_data['results']]
        class_counts = Counter(class_ids)

        quantity_qf = class_counts.get(0.0, 0)

        print(f"ОБРАБОТКА В recognize_only:")
        print(f"Найдено автоматов в text-service: {len(list_for_searching)}")
        print(f"Ожидается по детекции: {quantity_qf}")
        print(f"Типы монтажа из draw_rect: {mounting_types}")

        processed_qf_list = list(list_for_searching)

        # Создание пустых автоматов, если текст не найден
        if quantity_qf > len(processed_qf_list):
            actual_quantity = quantity_qf - len(processed_qf_list)
            print(f"  - Создаем {actual_quantity} пустых QF объектов")

            for i in range(actual_quantity):
                auto_number = len(processed_qf_list) + i + 1
                mounting_type = mounting_types.get(auto_number, "0")

                empty_qf = qf.QF(
                    ID_QF="",
                    Name="",
                    Polus="",
                    Current="",
                    Voltage="",
                    Current_Close="",
                    Mounting_Type=str(mounting_type) if mounting_type is not None else "0"
                )
                processed_qf_list.append(empty_qf)

        count_transformators, count_wh = split_by_element_type(class_counts)

        # Преобразование объектов QF в словари для API
        detected_elements = []

        # 1. Автоматические выключатели (QF)
        for i, qf_obj in enumerate(processed_qf_list):
            # Преобразуем объект QF в словарь
            detected_elements.append({
                "id": f"QF_{i+1}",
                "type": "QF",
                "number": i + 1,
                "confidence": 0.9,
                "parameters": {
                    "current": str(qf_obj.Current) if hasattr(qf_obj, 'Current') and qf_obj.Current is not None else "",
                    "voltage": str(qf_obj.Voltage) if hasattr(qf_obj, 'Voltage') and qf_obj.Voltage is not None else "",
                    "current_close": str(qf_obj.Current_Close) if hasattr(qf_obj, 'Current_Close') and qf_obj.Current_Close is not None else "",
                    "mounting_type": str(qf_obj.Mounting_Type) if hasattr(qf_obj, 'Mounting_Type') and qf_obj.Mounting_Type is not None else "",
                    "name": str(qf_obj.Name) if hasattr(qf_obj, 'Name') and qf_obj.Name is not None else "",
                    "polus": str(qf_obj.Polus) if hasattr(qf_obj, 'Polus') and qf_obj.Polus is not None else "",
                    "id_qf": str(qf_obj.ID_QF) if hasattr(qf_obj, 'ID_QF') and qf_obj.ID_QF is not None else ""
                }
            })

        # 2. Трансформаторы
        transformator_count = sum(count_transformators) if count_transformators else 0
        if transformator_count > 0:
            for i in range(transformator_count):
                detected_elements.append({
                    "id": f"TR_{i+1}",
                    "type": "transformer",
                    "number": len(detected_elements) + 1,
                    "confidence": 0.9,
                    "parameters": {
                        "quantity": "1",
                        "power": "",
                        "voltage_in": "",
                        "voltage_out": ""
                    }
                })

        # 3. Счетчики
        counter_count = sum(count_wh) if count_wh else 0
        if counter_count > 0:
            for i in range(counter_count):
                detected_elements.append({
                    "id": f"CNT_{i+1}",
                    "type": "counter",
                    "number": len(detected_elements) + 1,
                    "confidence": 0.9,
                    "parameters": {
                        "type": "электрический счетчик",
                        "phase": "3-фазный"
                    }
                })

        # Добавление элементов из list_for_quantity (трансформаторы с параметрами)
        for i, ta_obj in enumerate(list_for_quantity):
            if hasattr(ta_obj, 'ta_quantity'):
                for j in range(ta_obj.ta_quantity):
                    detected_elements.append({
                        "id": f"TA_{i+1}_{j+1}",
                        "type": "transformer",
                        "number": len(detected_elements) + 1,
                        "confidence": 0.9,
                        "parameters": {
                            "quantity": str(ta_obj.ta_quantity),
                            "ta_name": str(ta_obj.ta_name)
                        }
                    })

        print(f"Распознано элементов: {len(detected_elements)}")
        print(f" - Автоматов: {len([e for e in detected_elements if e['type'] == 'QF'])}")
        print(f" - Трансформаторов: {len([e for e in detected_elements if e['type'] == 'transformer'])}")
        print(f" - Счетчиков: {len([e for e in detected_elements if e['type'] == 'counter'])}")

        # Информация о QF объектах для отладки
        qf_elements = [e for e in detected_elements if e['type'] == 'QF']
        print(f"\nДетали QF элементов:")
        for qf_elem in qf_elements:
            params = qf_elem['parameters']
            print(f"  - QF {qf_elem['id']}: ток='{params.get('current')}', напряжение='{params.get('voltage')}', "
                  f"монтаж='{params.get('mounting_type')}'")

        return image, detected_elements

    except Exception as e:
        print(f"Error processing image in recognize_only: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def analyze_circuit_diagram(results, image):
    """
    Анализирует результаты детекции и строит структурированное описание схемы.

    Шаги:
        1. Разбирает результаты детекции на группы bbox (текст, автоматы,
           трансформаторы, УЗО, bracing, прочее) через results_boxes.
        2. Создает нумерацию текстовых блоков без распознавания
           (create_text_numbers).
        3. Строит граф схемы (create_graphs) и карты соответствий
           текст ↔ элемент (create_a_mind_map).
        4. Определяет тип монтажа для каждого автомата
           (determine_mounting_type_and_bbox) с учетом bracing.
        5. Нумерует автоматы с учетом типа монтажа и наличия текста
           (numeric_automatics_with_mounting), затем остальные элементы
           (numeric_other_elements).
        6. Формирует связанные текстовые чанки (create_connect_text_chunks)
           и отправляет их в text-service (recognize_with_text_service).
        7. Рисует визуализацию схемы (render_circuit_visualization).

    Args:
        results (list[dict]): Результаты детекции в формате
            [{'results': [...], 'names': {...}}].
        image (np.ndarray): Исходное изображение (OpenCV BGR).

    Returns:
        tuple[list, list, dict]:
            - list_for_searching: список QF-объектов для поиска в БД.
            - list_for_quantity: список TA-объектов (трансформаторы тока)
              с вычисленным количеством.
            - mounting_types_dict: словарь {номер_автомата: тип_монтажа}.
    """
    thickness = 2
    font_scale = 0.8
    SCORE_THRESHOLD = 0.5

    circuit_graph = cg.CircuitGraph()

    text_boxes, automatics_boxes, other_boxes, transformer_boxes, rcd_boxes, bracing_boxes = results_boxes(results,
                                                                                                           SCORE_THRESHOLD)

    # ШАГ 1: Создание нумерации текстовых блоков (без распознавания)
    text_boxes_sorted, text_numbers = create_text_numbers(text_boxes)

    node_ids, text_nodes = create_graphs(circuit_graph, automatics_boxes, transformer_boxes, rcd_boxes, text_boxes)

    text_to_automatic_map, automatic_to_text_map, maps_by_type = create_a_mind_map(
        circuit_graph, text_boxes, automatics_boxes, node_ids, text_nodes,
        transformer_boxes=transformer_boxes, rcd_boxes=rcd_boxes
    )

    # ШАГ 2: Определение типа монтажа
    print("ОПРЕДЕЛЕНИЕ ТИПОВ МОНТАЖА ДЛЯ АВТОМАТИЧЕСКИХ ВЫКЛЮЧАТЕЛЕЙ...")
    automatic_mounting_data = {}
    mounting_types_dict = {}

    for i, auto_box in enumerate(automatics_boxes):
        mounting_type, final_bbox = determine_mounting_type_and_bbox(auto_box, bracing_boxes)
        automatic_mounting_data[auto_box] = (mounting_type, final_bbox)

    # ШАГ 3: Нумерация автоматов с использованием карт автоматов (maps_by_type['automatic'])
    # Это гарантирует, что тексты трансформаторов не привяжутся к автоматам
    auto_text_map = maps_by_type['automatic']['text_to_elem']

    box_numbers, automatics_with_texts, current_number = numeric_automatics_with_mounting(
        text_boxes_sorted, auto_text_map, automatics_boxes,
        text_numbers, automatic_mounting_data
    )

    for auto_box, auto_number in box_numbers.items():
        mounting_type, _ = automatic_mounting_data[auto_box]
        mounting_types_dict[auto_number] = mounting_type

    all_other_elements_sorted = numeric_other_elements(
        transformer_boxes, rcd_boxes, other_boxes, box_numbers, current_number
    )

    # ШАГ 4: Создаем связанные текстовые чанки
    connected_text_chunks = create_connect_text_chunks(
        circuit_graph, automatics_with_texts, automatic_to_text_map, box_numbers,
        node_ids, text_nodes, text_boxes, text_numbers, automatic_mounting_data
    )

    total_texts = len(text_boxes)
    connected_texts = len(connected_text_chunks)
    print(f"ФИЛЬТРАЦИЯ: {connected_texts}/{total_texts} текстов связаны с элементами и будут отправлены в БД")

    # ШАГ 5: Отправление связных текстов на распознавание
    text_service_url = os.getenv("TEXT_SERVICE_URL", "http://text-processing-service:5002")
    list_for_searching, list_for_quantity = recognize_with_text_service(
        connected_text_chunks, image, text_boxes, text_numbers, text_service_url
    )

    # ШАГ 6: Визуализация
    print("Визуализация связей и отрисовываю элементы...")

    render_circuit_visualization(
        image=image,
        automatics_boxes=automatics_boxes,
        automatic_mounting_data=automatic_mounting_data,
        all_other_elements_sorted=all_other_elements_sorted,
        text_boxes_sorted=text_boxes_sorted,
        text_numbers=text_numbers,
        box_numbers=box_numbers,
        circuit_graph=None,  # только при тестировании (circuit_graph)
        thickness=thickness,
        font_scale=font_scale
    )

    return list_for_searching, list_for_quantity, mounting_types_dict


def results_boxes(results, SCORE_THRESHOLD):
    """
    Разбирает результаты детекции YOLO и распределяет bbox по типам элементов.

    Поддерживает два формата результатов:
        - dict с ключами 'results' и 'names';
        - объект YOLO с атрибутами .boxes.data и .names.

    Для каждого bbox:
        - Отбрасывает детекции с conf < SCORE_THRESHOLD.
        - Формирует кортеж (x1, y1, x2, y2, conf, cls_id, cx, cy).
        - Кладет в соответствующий список по cls_id.

    Args:
        results (list): Список результатов детекции (dict или объект YOLO).
        SCORE_THRESHOLD (float): Минимальная уверенность для учета bbox.

    Returns:
        tuple[list, list, list, list, list, list]:
            - text_boxes: bbox текстовых блоков (cls_id = 7).
            - automatics_boxes: bbox автоматических выключателей (cls_id = 0).
            - other_boxes: bbox прочих элементов.
            - transformer_boxes: bbox трансформаторов (cls_id = 1).
            - rcd_boxes: bbox УЗО (cls_id = 9).
            - bracing_boxes: bbox bracing-элементов (cls_id = 4).
    """
    text_boxes = []
    automatics_boxes = []
    other_boxes = []
    transformer_boxes = []
    rcd_boxes = []
    bracing_boxes = []

    print(f"АНАЛИЗ РЕЗУЛЬТАТОВ ДЕТЕКЦИИ:")

    for result in results:
        if isinstance(result, dict):
            boxes_data = result['results']
            names = result['names']
            print(f"Результат как dict: {len(boxes_data)} боксов")
            print(f"Имена классов: {names}")
        else:
            boxes_data = result.boxes.data.cpu().numpy()
            names = result.names
            print(f"Результат как объект: {len(boxes_data)} боксов")
            print(f"Имена классов: {names}")

        for i, box in enumerate(boxes_data):
            x1, y1, x2, y2, conf, cls_id = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf = float(conf)
            cls_id = int(cls_id)

            if conf < SCORE_THRESHOLD:
                continue

            box_data = (x1, y1, x2, y2, conf, cls_id, (x1 + x2) / 2, (y1 + y2) / 2)

            if cls_id == YOLOClassID.TEXT:
                text_boxes.append(box_data)
                print(f"Текст {len(text_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            elif cls_id == YOLOClassID.AUTOMATIC:
                automatics_boxes.append(box_data)
                print(f"Automatic {len(automatics_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            elif cls_id == YOLOClassID.TRANSFORMATOR:
                transformer_boxes.append(box_data)
                print(f"Transformer {len(transformer_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            elif cls_id == YOLOClassID.RCD:
                rcd_boxes.append(box_data)
                print(f"RCD {len(rcd_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            elif cls_id == YOLOClassID.BRACING:
                bracing_boxes.append(box_data)
                print(f"BRACING {len(bracing_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            else:
                other_boxes.append(box_data)
                print(f"Other {len(other_boxes)} (cls_id={cls_id}): bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")

    print(f"ИТОГО ОБНАРУЖЕНО:")
    print(f"- Automatics: {len(automatics_boxes)}")
    print(f"- Bracing: {len(bracing_boxes)}")
    print(f"- Text: {len(text_boxes)}")
    print(f"- Transformers: {len(transformer_boxes)}")
    print(f"- RCD: {len(rcd_boxes)}")
    print(f"- Other: {len(other_boxes)}")

    return text_boxes, automatics_boxes, other_boxes, transformer_boxes, rcd_boxes, bracing_boxes


def numeric_automatics_with_mounting(text_boxes_sorted, text_to_automatic_map,
                                     automatics_boxes, text_numbers, automatic_mounting_data):
    """
    Нумерует автоматические выключатели с учетом типа монтажа.

    Логика:
        1. Идет по отсортированным текстовым блокам и для каждого текста,
           у которого есть связанный автомат (text_to_automatic_map),
           присваивает автомату следующий номер.
        2. Автоматы без связанного текста нумеруются после всех
           «текстовых» автоматов в порядке исходного списка automatics_boxes.

    Args:
        text_boxes_sorted (list): Отсортированные bbox текстовых блоков.
        text_to_automatic_map (dict): Карта {text_bbox_tuple: auto_bbox}.
        automatics_boxes (list): Список bbox всех автоматов.
        text_numbers (dict): Карта {text_bbox: номер_текста}.
        automatic_mounting_data (dict): Карта
            {auto_bbox: (mounting_type, final_bbox)}.

    Returns:
        tuple[dict, list, int]:
            - box_numbers: карта {auto_bbox: номер_автомата}.
            - automatics_with_texts: список bbox автоматов (с текстом
              и без), в порядке нумерации.
            - current_number: следующий свободный номер (после всех
              пронумерованных автоматов).
    """
    box_numbers = {}
    current_number = 1

    print(" Нумерация автоматов с определением типа монтажа:")

    used_automatics = set()
    automatics_with_texts = []

    for text_box in text_boxes_sorted:
        text_key = tuple(text_box)
        if text_key in text_to_automatic_map:
            auto_box = text_to_automatic_map[text_key]

            if auto_box not in box_numbers:
                mounting_type, final_bbox = automatic_mounting_data[auto_box]

                box_numbers[auto_box] = current_number
                used_automatics.add(auto_box)
                automatics_with_texts.append(auto_box)

                # Берём сохранённый номер текста
                text_num = text_numbers.get(text_key, "?")
                print(f"Автомат {current_number} (монтаж тип {mounting_type}) → Текст №{text_num}")
                current_number += 1

    remaining_automatics = [auto for auto in automatics_boxes if auto not in used_automatics]
    for auto_box in remaining_automatics:
        mounting_type, final_bbox = automatic_mounting_data[auto_box]

        box_numbers[auto_box] = current_number
        automatics_with_texts.append(auto_box)
        print(f"Автомат {current_number} (монтаж тип {mounting_type}, без связанного текста)")
        current_number += 1

    return box_numbers, automatics_with_texts, current_number


def numeric_other_elements(transformer_boxes, rcd_boxes, other_boxes, box_numbers, current_number):
    """
    Нумерует все «остальные» элементы (трансформаторы, УЗО, прочее)
    после автоматов, совместно.

    Все остальные элементы объединяются в один список, сортируются
    по положению (Y, затем X) и получают сквозные номера, начиная
    с current_number.

    Args:
        transformer_boxes (list): Bbox трансформаторов.
        rcd_boxes (list): Bbox УЗО.
        other_boxes (list): Bbox прочих элементов.
        box_numbers (dict): Карта {bbox: номер}, дополняется на месте.
        current_number (int): Следующий свободный номер.

    Returns:
        list: Отсортированный список всех остальных элементов.
    """
    # ШАГ 3: Нумерация ВСЕХ остальных элементов совместно (после автоматов)
    print("  Остальные элементы (совместная нумерация):")

    # Объединяем все остальные классы в один список
    all_other_elements = transformer_boxes + rcd_boxes + other_boxes

    # Сортируем все остальные элементы по положению (Y, затем X)
    all_other_elements_sorted = sorted(all_other_elements, key=lambda b: (b[7], b[6]))

    for element in all_other_elements_sorted:
        box_numbers[element] = current_number

        # Определяем тип элемента для лога
        if element in transformer_boxes:
            element_type = "трансформатор"
        elif element in rcd_boxes:
            element_type = "УЗО"
        else:
            element_type = "другой элемент"

        print(f"{element_type} -> номер {current_number}")
        current_number += 1
    return all_other_elements_sorted


def create_text_numbers(text_boxes):
    """
    Создает нумерацию текстовых блоков без распознавания текста.

    Сортирует текстовые блоки в порядке чтения (сверху вниз, слева направо)
    и присваивает каждому порядковый номер.

    Args:
        text_boxes (list): Список bbox текстовых блоков.

    Returns:
        tuple[list, dict]:
            - text_boxes_sorted: отсортированный список bbox.
            - text_numbers: карта {bbox: номер_текста}.
    """
    # Сортируем текстовые блоки в порядке чтения (слева направо, сверху вниз)
    text_boxes_sorted = sorted(text_boxes, key=lambda b: (b[7], b[6]))  # Y, затем X

    # Создаем словарь для нумерации
    text_numbers = {}
    for idx, text in enumerate(text_boxes_sorted, 1):
        text_numbers[text] = idx

    return text_boxes_sorted, text_numbers


def split_by_element_type(class_counts):
    """
    Разделяет подсчет классов на трансформаторы и счетчики.

    Args:
        class_counts (Counter): Счетчик {class_id: количество}.

    Returns:
        tuple[list, list]:
            - list_trans: [количество трансформаторов (class_id = 1)].
            - list_wh: [количество счетчиков (class_id = 2)].
    """
    count_transformators = class_counts.get(1.0, 0)
    count_wh = class_counts.get(2.0, 0)

    list_trans = []
    list_wh = []

    list_trans.append(count_transformators)
    list_wh.append(count_wh)
    return list_trans, list_wh


def print_search_results(request_results, request_trans_results, request_wh_results):
    """
    Выводит результаты поиска в консоль.

    Args:
        request_results (list): Найденные автоматические выключатели.
        request_trans_results (list): Найденные трансформаторы тока
            (список списков).
        request_wh_results (list): Найденные счетчики тока
            (список списков).

    Returns:
        None
    """
    if len(request_results) != 0:
        print('Найденные автоматические выключатели: ')
        for index, elements in enumerate(request_results, 1):
            print('Позиция', index)
            print(elements)
            print('\n')

    if len(request_trans_results) != 0:
        print('Найденные трансформаторы тока: ')
        for index, elements_trans in enumerate(request_trans_results, 1):
            print('Позиция', index)
            for elem in elements_trans:
                print(elem)
            print('\n')

    if len(request_wh_results) != 0:
        print('Найденные счетчики тока: ')
        for index, elements_wh in enumerate(request_wh_results, 1):
            print('Позиция', index)
            for elem in elements_wh:
                print(elem)
            print('\n')



