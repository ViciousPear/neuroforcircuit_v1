import requests
import cv2
from collections import Counter
# from db_api import db_client_api
import db_client_api
import os
import qf, ta
# from . import qf, ta
import search_text
# from . import search_text
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# from . import circuit_graph as cg
import circuit_graph as cg

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

def process_text_regions(text_boxes, image, text_service_url):
    """Отправляет на распознавание текста части изображения"""
    list_for_searching = []
    list_for_quantity = []
    
    # Подготавливаем текстовые чанки для отправки
    text_chunks = []
    
    # Создаем сессию с retry логикой
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Обработка текста
    for text in text_boxes:
        x1, y1, x2, y2, conf, cls_id, cx, cy = text
        color = colors[cls_id % len(colors)]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
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
            response = session.post(
                f"{text_service_url}/api/process-text-batch", 
                json={
                    "chunks": text_chunks,
                    "image_size": [image.shape[1], image.shape[0]],
                    "image_shape": list(image.shape)
                },
                timeout=300 
            )
            response.raise_for_status()
            
            # Обрабатываем результаты
            batch_results = response.json()
            
            for result in batch_results:
                if result.get("type") == "circuit_breaker":
                    qf_dict = result.get("object", {})
                    qf_obj = qf.create_qf(
                        qf_dict.get("ID_QF", ""),
                        qf_dict.get("Name", ""),
                        qf_dict.get("Current", ""),
                        qf_dict.get("Voltage", ""),
                        qf_dict.get("Current_Close", ""),
                        qf_dict.get("Polus", ""),
                        qf_dict.get("Mounting_Type", "")
                    )
                    list_for_searching.append(qf_obj)
                    
                elif result.get("type") == "current_transformer":
                    ta_dict = result.get("object", {})
                    # Исправляем получение имени для приватного поля
                    ta_name = ta_dict.get("_Trans_TA__ta_name", "")
                    ta_obj = ta.create_ta(ta_name)
                    list_for_quantity.append(ta_obj)
                    
        except requests.exceptions.Timeout:
            print("⚠️ Timeout after 300 seconds! Using fallback...")
            # Fallback логика
        except requests.exceptions.RetryError:
            print("⚠️ All retries failed! Using fallback...")
            # Fallback логика
        except Exception as e:
            print(f"❌ Unexpected error: {e}")            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса к text-service: {e}")
    
    return list_for_searching, list_for_quantity


def draw_rect(results, image):
    thickness = 2
    font_scale = 0.8
    SCORE_THRESHOLD = 0.5

    circuit_graph = cg.CircuitGraph()

    text_boxes, automatics_boxes, other_boxes, transformer_boxes, rcd_boxes, bracing_boxes = results_boxes(results, SCORE_THRESHOLD)

    # ШАГ 1: Создаем нумерацию текстовых блоков (без распознавания)
    text_boxes_sorted, text_numbers = create_text_numbers(text_boxes)

    # Создаем графы и получаем маппинги
    node_ids, text_nodes = create_graphs(circuit_graph, automatics_boxes, transformer_boxes, rcd_boxes, text_boxes)

    # Создаем маппинги между текстами и автоматами
    text_to_automatic_map, automatic_to_text_map = create_a_mind_map(circuit_graph, text_boxes, automatics_boxes, node_ids, text_nodes)

    # ШАГ 2: Определяем типы монтажа
    print("🔧 ОПРЕДЕЛЕНИЕ ТИПОВ МОНТАЖА ДЛЯ АВТОМАТИЧЕСКИХ ВЫКЛЮЧАТЕЛЕЙ...")
    automatic_mounting_data = {}
    mounting_types_dict = {}
    
    for i, auto_box in enumerate(automatics_boxes):
        mounting_type, final_bbox = determine_mounting_type_and_bbox(auto_box, bracing_boxes)
        automatic_mounting_data[auto_box] = (mounting_type, final_bbox)

    # ШАГ 3: Нумеруем автоматы с использованием сохраненной нумерации текстов
    box_numbers, automatics_with_texts, current_number = numeric_automatics_with_mounting(
        text_boxes_sorted, text_to_automatic_map, text_nodes, automatics_boxes, 
        text_numbers, automatic_mounting_data  # Передаем text_numbers вместо recognized_texts
    )

    # Заполняем mounting_types_dict
    for auto_box, auto_number in box_numbers.items():
        mounting_type, _ = automatic_mounting_data[auto_box]
        mounting_types_dict[auto_number] = mounting_type

    # Нумеруем остальные элементы
    all_other_elements_sorted = numeric_other_elements(
        transformer_boxes, rcd_boxes, other_boxes, box_numbers, current_number
    )

    # ШАГ 4: Создаем связанные текстовые чанки (без распознавания текста)
    connected_text_chunks = create_connect_text_chunks(
        circuit_graph, automatics_with_texts, automatic_to_text_map, box_numbers, 
        node_ids, text_nodes, text_boxes, text_numbers, automatic_mounting_data
    )

    # ЛОГИРУЕМ РЕЗУЛЬТАТЫ ФИЛЬТРАЦИИ
    total_texts = len(text_boxes)
    connected_texts = len(connected_text_chunks)
    print(f"📊 ФИЛЬТРАЦИЯ: {connected_texts}/{total_texts} текстов связаны с элементами и будут отправлены в БД")

    # ШАГ 5: Отправляем на распознавание ТОЛЬКО связанные тексты
    text_service_url = os.getenv("TEXT_SERVICE_URL", "http://text-processing-service:5002")
    # локальный: http://localhost:500
    # для тестов: http://82.202.129.245:5002
    # продакшн: http://text-processing-service:5002
    list_for_searching, list_for_quantity = recognize_with_text_service(
        connected_text_chunks, image, text_boxes, text_numbers, text_service_url
    )

    # ШАГ 6: Визуализация (используем сохраненную нумерацию)
    print("🎨 Визуализирую связи и отрисовываю элементы...")

    # visualize_connections(circuit_graph, image)
    visualize_automatics_with_mounting(
        image, automatics_boxes, automatic_mounting_data, box_numbers, thickness, font_scale
    )
    visualize_other_elements(
        image, all_other_elements_sorted, box_numbers, thickness, font_scale
    )

    # Визуализация текстовых блоков с сохраненной нумерацией
    print("📝 Выделяю текстовые блоки...")
    visualize_rectangles_for_texts(
        image, text_boxes_sorted, text_nodes, text_numbers, thickness
    )

    return list_for_searching, list_for_quantity, mounting_types_dict


def results_boxes(results, SCORE_THRESHOLD):
    text_boxes = []
    automatics_boxes = []
    other_boxes = []
    transformer_boxes = []
    rcd_boxes = []
    bracing_boxes = []  # Новый список для объектов крепления
    
    print(f"🔍 АНАЛИЗ РЕЗУЛЬТАТОВ ДЕТЕКЦИИ:")
    
    for result in results:
        if isinstance(result, dict):
            boxes_data = result['results']
            names = result['names']
            print(f"  📊 Результат как dict: {len(boxes_data)} боксов")
            print(f"  📊 Имена классов: {names}")
        else:
            boxes_data = result.boxes.data.cpu().numpy()
            names = result.names
            print(f"  📊 Результат как объект: {len(boxes_data)} боксов")
            print(f"  📊 Имена классов: {names}")

        for i, box in enumerate(boxes_data):
            x1, y1, x2, y2, conf, cls_id = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf = float(conf)
            cls_id = int(cls_id)

            if conf < SCORE_THRESHOLD:
                continue

            box_data = (x1, y1, x2, y2, conf, cls_id, (x1 + x2) / 2, (y1 + y2) / 2)

            if cls_id == 8:  # text
                text_boxes.append(box_data)
                print(f"    📝 Текст {len(text_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            elif cls_id == 0:  # automatics
                automatics_boxes.append(box_data)
                print(f"    🔵 Automatic {len(automatics_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            elif cls_id == 1:  # transformers
                transformer_boxes.append(box_data)
                print(f"    🔴 Transformer {len(transformer_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            elif cls_id == 10:  # УЗО
                rcd_boxes.append(box_data)
                print(f"    🟣 RCD {len(rcd_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            elif cls_id == 4:  # bracing 
                bracing_boxes.append(box_data)
                print(f"    🟡 BRACING {len(bracing_boxes)}: bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
            else:
                other_boxes.append(box_data)
                print(f"    ⚪ Other {len(other_boxes)} (cls_id={cls_id}): bbox({x1}, {y1}, {x2}, {y2}), conf={conf:.2f}")
    
    print(f"🎯 ИТОГО ОБНАРУЖЕНО:")
    print(f"  - Automatics: {len(automatics_boxes)}")
    print(f"  - Bracing: {len(bracing_boxes)}")
    print(f"  - Text: {len(text_boxes)}")
    print(f"  - Transformers: {len(transformer_boxes)}")
    print(f"  - RCD: {len(rcd_boxes)}")
    print(f"  - Other: {len(other_boxes)}")
                
    return text_boxes, automatics_boxes, other_boxes, transformer_boxes, rcd_boxes, bracing_boxes

def create_graphs(circuit_graph, automatics_boxes, transformer_boxes, rcd_boxes, text_boxes):
    node_ids = {}
    text_nodes = {}
    
    # Добавляем автоматы в граф
    for auto in automatics_boxes:
        x1, y1, x2, y2, conf, cls_id, cx, cy = auto
        node_id = circuit_graph.add_node(
            bbox=[x1, y1, x2, y2],
            node_type='automatic',
            confidence=conf
        )
        node_ids[auto] = node_id

    # Добавляем трансформаторы
    for transformer in transformer_boxes:
        x1, y1, x2, y2, conf, cls_id, cx, cy = transformer
        node_id = circuit_graph.add_node(
            bbox=[x1, y1, x2, y2],
            node_type='transformer',
            confidence=conf
        )
        node_ids[transformer] = node_id

    # Добавляем УЗО
    for rcd in rcd_boxes:
        x1, y1, x2, y2, conf, cls_id, cx, cy = rcd
        node_id = circuit_graph.add_node(
            bbox=[x1, y1, x2, y2],
            node_type='rcd',
            confidence=conf
        )
        node_ids[rcd] = node_id

    # Добавляем текстовые узлы
    for text in text_boxes:
        x1, y1, x2, y2, conf, cls_id, cx, cy = text
        node_id = circuit_graph.add_node(
            bbox=[x1, y1, x2, y2],
            node_type='text',
            confidence=conf
        )
        text_nodes[text] = node_id

    # СОЗДАЕМ АДАПТИВНЫЕ СВЯЗИ В ГРАФЕ
    print("🔗 Создаю АДАПТИВНЫЕ связи в графе...")

    # Для трансформаторов используем больший коэффициент, т.к. они обычно крупнее
    # Можно создать отдельный граф с другим коэффициентом или изменить логику
    # Все элементы используют один и тот же коэффициент (1.7)
    circuit_graph.create_exclusive_connections_adaptive('automatic')
    circuit_graph.create_exclusive_connections_adaptive('transformer')
    circuit_graph.create_exclusive_connections_adaptive('rcd')

    print(f"✅ Граф построен: {len(circuit_graph.nodes)} узлов, {len(circuit_graph.edges)} связей")
    return node_ids, text_nodes

def recognize_text(circuit_graph, text_boxes, image, text_nodes):
    """Сортируем текстовые блоки и сохраняем геометрию, БЕЗ распознавания текста"""
    # НУМЕРАЦИЯ ОБЪЕКТОВ НА ИЗОБРАЖЕНИИ (ДО отправки в text-service)
    print("🔢 Назначаю номера объектам на изображении...")

    # Сортируем текстовые блоки в порядке чтения (слева направо, сверху вниз)
    text_boxes_sorted = sorted(text_boxes, key=lambda b: (b[7], b[6]))  # Y, затем X
    
    # Сохраняем геометрию в граф (без распознавания текста)
    for text in text_boxes:
        x1, y1, x2, y2, conf, cls_id, cx, cy = text
        text_node_id = text_nodes[text]
        circuit_graph.nodes[text_node_id]['bbox'] = [x1, y1, x2, y2]
        circuit_graph.nodes[text_node_id]['center'] = (cx, cy)
        circuit_graph.nodes[text_node_id]['confidence'] = conf
    
    return text_boxes_sorted, {}  # Возвращаем пустой словарь recognized_texts

def create_a_mind_map(circuit_graph, text_boxes, automatics_boxes, node_ids, text_nodes):
    text_to_automatic_map = {}
    automatic_to_text_map = {}

    # Собираем связи между текстами и автоматами из графа
    for edge in circuit_graph.edges:
        node1_id, node2_id, connection_type, distance = edge
        node1 = circuit_graph.nodes[node1_id]
        node2 = circuit_graph.nodes[node2_id]

        # Находим связи текст-автомат
        if (node1['type'] == 'text' and node2['type'] == 'automatic'):
            text_node = node1
            auto_node = node2
        elif (node1['type'] == 'automatic' and node2['type'] == 'text'):
            text_node = node2
            auto_node = node1
        else:
            continue

        # Находим соответствующие боксы
        text_box = None
        auto_box = None

        for text in text_boxes:
            if text_nodes.get(text) == text_node['id']:
                text_box = text
                break

        for auto in automatics_boxes:
            if node_ids.get(auto) == auto_node['id']:
                auto_box = auto
                break

        if text_box and auto_box:
            text_to_automatic_map[tuple(text_box)] = auto_box
            automatic_to_text_map[tuple(auto_box)] = text_box

    return text_to_automatic_map, automatic_to_text_map

def determine_mounting_type_and_bbox(automatic_box, bracing_boxes, max_distance=40):
    """Определяет тип монтажа и возвращает объединенный bbox если тип = 1"""
    x1_a, y1_a, x2_a, y2_a, conf_a, cls_id_a, cx_a, cy_a = automatic_box
    
    print(f"  🔧 Анализ автомата: bbox({x1_a}, {y1_a}, {x2_a}, {y2_a})")
    print(f"  🔧 Всего bracing объектов: {len(bracing_boxes)}")
    
    top_bracings, bottom_bracings = find_nearby_bracing(automatic_box, bracing_boxes, max_distance)
    
    # Если есть хотя бы один bracing сверху и хотя бы один снизу
    if len(top_bracings) >= 1 and len(bottom_bracings) >= 1:
        # Находим крайние точки среди всех связанных bracing и automatic
        all_related_boxes = [automatic_box] + top_bracings + bottom_bracings
        
        min_x = min(box[0] for box in all_related_boxes)
        min_y = min(box[1] for box in all_related_boxes)
        max_x = max(box[2] for box in all_related_boxes)
        max_y = max(box[3] for box in all_related_boxes)
        
        print(f"    ✅ Монтаж тип 1: объединенный bbox({min_x}, {min_y}, {max_x}, {max_y})")
        return "1", (min_x, min_y, max_x, max_y)
    else:
        print(f"    ❌ Монтаж тип 0: недостаточно bracing (сверху: {len(top_bracings)}, снизу: {len(bottom_bracings)})")
        return "0", automatic_box[:4]
    
def find_nearby_bracing(automatic_box, bracing_boxes, max_distance=45):
    """Находит объекты bracing сверху и снизу от automatic_box в пределах max_distance"""
    x1_a, y1_a, x2_a, y2_a, conf_a, cls_id_a, cx_a, cy_a = automatic_box
    
    top_bracings = []
    bottom_bracings = []
    
    print(f"    🔍 Поиск bracing для автомата: Y=[{y1_a}-{y2_a}], высота={y2_a-y1_a}px")
    
    for i, bracing in enumerate(bracing_boxes):
        x1_b, y1_b, x2_b, y2_b, conf_b, cls_id_b, cx_b, cy_b = bracing
        
        print(f"      🔎 Анализ bracing {i+1}: Y=[{y1_b}-{y2_b}], высота={y2_b-y1_b}px")
        
        # Проверяем горизонтальное перекрытие (должно быть значительным)
        horizontal_overlap = calculate_horizontal_overlap(automatic_box, bracing)
        
        if horizontal_overlap > 0.3:
            # Вычисляем расстояние от границ и определяем положение
            vertical_distance, is_above = calculate_vertical_distance_from_edges(automatic_box, bracing)
            
            if vertical_distance <= max_distance and vertical_distance > 0:
                if is_above:  # bracing выше automatic
                    top_bracings.append(bracing)
                    print(f"      ✅ ДОБАВЛЕНО СВЕРХУ: расстояние={vertical_distance:.1f}px")
                else:  # bracing ниже automatic
                    bottom_bracings.append(bracing)
                    print(f"      ✅ ДОБАВЛЕНО СНИЗУ: расстояние={vertical_distance:.1f}px")
            else:
                print(f"      ❌ ПРОПУЩЕНО: расстояние {vertical_distance:.1f}px > {max_distance}px")
        else:
            print(f"      ❌ ПРОПУЩЕНО: слабое горизонтальное перекрытие {horizontal_overlap:.2f}")
    
    print(f"    🔍 Итог: {len(top_bracings)} сверху, {len(bottom_bracings)} снизу")
    return top_bracings, bottom_bracings

def calculate_vertical_distance_from_edges(automatic_box, bracing_box):
    """Вычисляет вертикальное расстояние от верхней/нижней границы automatic до ближайшей границы bracing"""
    x1_a, y1_a, x2_a, y2_a = automatic_box[:4]
    x1_b, y1_b, x2_b, y2_b = bracing_box[:4]
    
    # Расстояние от верхней границы automatic до нижней границы bracing (если bracing выше)
    distance_from_top = y1_a - y2_b if y2_b < y1_a else 0
    
    # Расстояние от нижней границы automatic до верхней границы bracing (если bracing ниже)
    distance_from_bottom = y1_b - y2_a if y1_b > y2_a else 0
    
    # Определяем положение bracing относительно automatics
    is_above = y2_b < y1_a  # bracing полностью выше automatics
    is_below = y1_b > y2_a  # bracing полностью ниже automatics
    
    if is_above:
        vertical_distance = distance_from_top
        print(f"        📏 Bracing ВЫШЕ: расстояние от верха automatics до низа bracing = {vertical_distance:.1f}")
    elif is_below:
        vertical_distance = distance_from_bottom
        print(f"        📏 Bracing НИЖЕ: расстояние от низа automatics до верха bracing = {vertical_distance:.1f}")
    else:
        vertical_distance = 0  # есть перекрытие по Y
        print(f"        📏 Bracing ПЕРЕКРЫВАЕТСЯ по вертикали")
    
    return vertical_distance, is_above


def calculate_horizontal_overlap(box1, box2):
    """Вычисляет горизонтальное перекрытие между двумя bounding box"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Горизонтальное перекрытие
    overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    width1 = x2_1 - x1_1
    width2 = x2_2 - x1_2
    
    # Процент перекрытия по ширине
    overlap_percentage = overlap_x / min(width1, width2) if min(width1, width2) > 0 else 0
    return overlap_percentage

# Новая функция для нумерации автоматов с учетом типа монтажа
def numeric_automatics_with_mounting(text_boxes_sorted, text_to_automatic_map, text_nodes, 
                                   automatics_boxes, recognized_texts, automatic_mounting_data):
    box_numbers = {}
    current_number = 1
    
    print("  📍 Нумерация автоматов с определением типа монтажа:")

    # ШАГ 1: Нумеруем автоматы, у которых есть связанные тексты (в порядке текстов)
    used_automatics = set()
    automatics_with_texts = []

    for text_box in text_boxes_sorted:
        if tuple(text_box) in text_to_automatic_map:
            auto_box = text_to_automatic_map[tuple(text_box)]

            if auto_box not in box_numbers:  # Защита от дублирования
                mounting_type, final_bbox = automatic_mounting_data[auto_box]
                
                box_numbers[auto_box] = current_number
                used_automatics.add(auto_box)
                automatics_with_texts.append(auto_box)

                text_content = recognized_texts.get(text_nodes[text_box], "Unknown")
                print(f"    ✅ Автомат {current_number} (монтаж тип {mounting_type}) → Текст: '{text_content}'")
                current_number += 1

    # ШАГ 2: Нумеруем оставшиеся автоматы (без связанных текстов)
    remaining_automatics = [auto for auto in automatics_boxes if auto not in used_automatics]
    for auto_box in remaining_automatics:
        mounting_type, final_bbox = automatic_mounting_data[auto_box]
        
        box_numbers[auto_box] = current_number
        automatics_with_texts.append(auto_box)
        print(f"    ⚠️ Автомат {current_number} (монтаж тип {mounting_type}, без связанного текста)")
        current_number += 1

    return box_numbers, automatics_with_texts, current_number

# def draw_combined_rectangle(image, automatic_box, top_bracings, bottom_bracings, color, number):
#     """Рисует объединенный прямоугольник для automatic с bracing сверху и снизу"""
#     x1_a, y1_a, x2_a, y2_a = automatic_box[:4]
    
#     # Находим крайние точки среди всех связанных bracing
#     all_related_boxes = [automatic_box] + top_bracings + bottom_bracings
    
#     min_x = min(box[0] for box in all_related_boxes)
#     min_y = min(box[1] for box in all_related_boxes)
#     max_x = max(box[2] for box in all_related_boxes)
#     max_y = max(box[3] for box in all_related_boxes)
    
#     # Рисуем объединенный прямоугольник
#     thickness = 2
#     cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color, thickness)
    
#     # Добавляем номер
#     font_scale = 0.8
#     label = str(number)
    
#     (text_width, text_height), _ = cv2.getTextSize(
#         label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
#     )
    
#     cv2.rectangle(
#         image,
#         (min_x, min_y - text_height - 10),
#         (min_x + text_width, min_y),
#         color,
#         -1
#     )
    
#     cv2.putText(
#         image,
#         label,
#         (min_x, min_y - 5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         (255, 255, 255),
#         thickness
#     )
    
#     return (min_x, min_y, max_x, max_y)

def numeric_other_elements(transformer_boxes, rcd_boxes, other_boxes, box_numbers, current_number):
    # ШАГ 3: Нумеруем ВСЕ остальные элементы совместно (после автоматов)
    print("  📍 Остальные элементы (совместная нумерация):")


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

        print(f"  🔄 {element_type} → номер {current_number}")
        current_number += 1
    return all_other_elements_sorted

def create_text_numbers(text_boxes_sorted, text_to_automatic_map, box_numbers, recognized_texts, text_nodes):
    text_numbers = {}
    print("  📍 Текстовые блоки:")
    for idx, text in enumerate(text_boxes_sorted, 1):
        text_numbers[text] = idx
        text_content = recognized_texts.get(text_nodes[text], "Unknown")

        # Находим связанный автомат для этого текста
        connected_auto = None
        if tuple(text) in text_to_automatic_map:
            connected_auto = text_to_automatic_map[tuple(text)]
            auto_number = box_numbers.get(connected_auto, "?")
            print(f"    📝 Текст {idx} → Автомат {auto_number}: '{text_content}'")
        else:
            print(f"    📝 Текст {idx} (без автомата): '{text_content}'")
    
    return text_numbers

# Модифицируем функцию recognize_with_text_service для передачи Mounting_Type
def recognize_with_text_service(connected_text_chunks, image, text_boxes, text_numbers, text_service_url=None):
    """Отправляет связанные тексты на распознавание с уже имеющейся нумерацией"""
    list_for_searching = []
    list_for_quantity = []
    
    if text_service_url is None:
        text_service_url = os.getenv("TEXT_SERVICE_URL", "http://text-processing-service:5002")

    # Создаем сессию для запросов
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Отправляем ТОЛЬКО связанные тексты в text-service
    if connected_text_chunks:
        try:
            print(f"📨 Отправляю {len(connected_text_chunks)} связанных текстов в text-service...")
            
            # Распознаем текст ДЛЯ КАЖДОГО связанного текста
            chunks_with_text = []
            for chunk in connected_text_chunks:
                x1, y1, x2, y2 = chunk["bbox"]
                # РАСПОЗНАЕМ ТЕКСТ ЗДЕСЬ! (только один раз)
                text_content = search_text.recognize_text_from_bbox(image, x1, y1, x2, y2)
                
                if text_content:
                    chunk_with_text = chunk.copy()
                    chunk_with_text["text"] = text_content
                    chunks_with_text.append(chunk_with_text)
                    
                    # Логируем с использованием уже имеющейся нумерации
                    auto_number = chunk.get("automatic_number", "")
                    text_number = chunk.get("text_number", "")
                    element_type = chunk.get("connected_element_type", "элемент")
                    print(f"  🔗 {element_type} {auto_number} (Текст {text_number}): распознан текст '{text_content[:50]}...'")
                else:
                    print(f"  ⚠️ Не удалось распознать текст для bbox {chunk['bbox']}")

            if not chunks_with_text:
                print("⚠️ Нет распознанных текстов для отправки")
                return list_for_searching, list_for_quantity

            response = session.post(
                f"{text_service_url}/api/process-text-batch",
                json={
                    "chunks": chunks_with_text,
                    "image_size": [image.shape[1], image.shape[0]],
                    "image_shape": list(image.shape)
                },
                timeout=300
            )
            response.raise_for_status()

            batch_results = response.json()

            # Обрабатываем результаты с синхронизированной нумерацией
            for i, result in enumerate(batch_results):
                chunk = chunks_with_text[i]
                text_content = chunk["text"]

                if result.get("type") == "circuit_breaker":
                    qf_dict = result.get("object", {})
                    
                    # Получаем тип монтажа из chunk
                    mounting_type = chunk.get("mounting_type", "0")
                    
                    qf_obj = qf.create_qf(
                        qf_dict.get("ID_QF", ""),
                        qf_dict.get("Name", ""),
                        qf_dict.get("Current", ""),
                        qf_dict.get("Voltage", ""),
                        qf_dict.get("Current_Close", ""),
                        qf_dict.get("Polus", ""),
                        mounting_type
                    )
                    list_for_searching.append(qf_obj)

                    automatic_number = chunk.get("automatic_number", "Unknown")
                    text_number = chunk.get("text_number", "Unknown")
                    print(f"  ✅ Автомат {automatic_number} (Текст {text_number}, монтаж {mounting_type}): '{text_content}' → {qf_dict.get('ID_QF', 'Unknown')}")

                elif result.get("type") == "current_transformer":
                    ta_dict = result.get("object", {})
                    ta_name = ta_dict.get("_Trans_TA__ta_name", "")
                    ta_obj = ta.create_ta(ta_name)
                    list_for_quantity.append(ta_obj)
                    print(f"  ✅ Трансформатор: '{text_content}' → {ta_name}")

        except Exception as e:
            print(f"❌ Ошибка при обработке текстов: {e}")

    else:
        print("ℹ️ Нет связанных текстов для обработки")
        # Fallback: обрабатываем все тексты, но с сохраненной нумерацией
        # Вам нужно будет адаптировать process_text_regions для использования text_numbers
    
    return list_for_searching, list_for_quantity

def create_text_numbers(text_boxes):
    """Создает нумерацию текстовых блоков (сортировка и индексация) без распознавания текста"""
    # Сортируем текстовые блоки в порядке чтения (слева направо, сверху вниз)
    text_boxes_sorted = sorted(text_boxes, key=lambda b: (b[7], b[6]))  # Y, затем X
    
    # Создаем словарь для нумерации
    text_numbers = {}
    for idx, text in enumerate(text_boxes_sorted, 1):
        text_numbers[text] = idx
    
    return text_boxes_sorted, text_numbers

def create_connect_text_chunks(circuit_graph, automatics_with_texts, automatic_to_text_map, 
                              box_numbers, node_ids, text_nodes, text_boxes, 
                              text_numbers, automatic_mounting_data):
    """
    Создает связанные текстовые чанки с использованием уже имеющейся нумерации.
    
    Эта функция анализирует связи между элементами (автоматами, трансформаторами, УЗО) 
    и текстовыми блоками в графе, создавая структурированные чанки для последующей
    обработки text-service. Текст НЕ распознается в этой функции, а добавляется позже.
    
    Args:
        circuit_graph: Граф с элементами и связями
        automatics_with_texts: Список автоматов, связанных с текстами
        automatic_to_text_map: Словарь сопоставления автомат → текст
        box_numbers: Словарь номеров элементов (автоматов, трансформаторов и т.д.)
        node_ids: Словарь сопоставления bounding box → ID узла в графе
        text_nodes: Словарь сопоставления текстового bbox → ID текстового узла
        text_boxes: Список всех текстовых bounding box
        text_numbers: Словарь нумерации текстовых блоков (уже отсортированных)
        automatic_mounting_data: Данные о типе монтажа для автоматов
    
    Returns:
        List[dict]: Список чанков со структурой для отправки в text-service
    """
    connected_text_chunks = []

    # ШАГ 1: Собираем ВСЕ текстовые узлы, связанные с любыми элементами
    connected_text_nodes = set()
    
    print("🔍 Анализирую связи в графе для фильтрации текстов...")
    
    for edge in circuit_graph.edges:
        node1_id, node2_id, connection_type, distance = edge
        node1 = circuit_graph.nodes[node1_id]
        node2 = circuit_graph.nodes[node2_id]

        # Определяем связи "элемент-текст"
        # Вариант 1: узел1 - элемент, узел2 - текст
        if (node1['type'] != 'text' and node2['type'] == 'text'):
            text_node = node2
            connected_text_nodes.add(text_node['id'])
            
        # Вариант 2: узел1 - текст, узел2 - элемент
        elif (node1['type'] == 'text' and node2['type'] != 'text'):
            text_node = node1
            connected_text_nodes.add(text_node['id'])
    
    print(f"  📍 Найдено {len(connected_text_nodes)} текстовых узлов, связанных с элементами")

    # ШАГ 2: Создаем чанки для автоматов
    print("🔄 Создаю чанки для автоматов...")
    
    for auto_box in automatics_with_texts:
        # Проверяем, есть ли у этого автомата связанный текст
        if tuple(auto_box) in automatic_to_text_map:
            text_box = automatic_to_text_map[tuple(auto_box)]
            
            # Получаем номер текста из уже созданной нумерации
            text_number = text_numbers.get(text_box)
            
            if text_number is None:
                print(f"    ⚠️ Пропускаю автомат: не найден номер текста для bbox {text_box[:4]}")
                continue
                
            # Получаем номер автомата
            auto_number = box_numbers.get(auto_box)
            if auto_number is None:
                print(f"    ⚠️ Пропускаю автомат: не найден номер для bbox {auto_box[:4]}")
                continue
            
            # Получаем данные о монтаже
            if auto_box not in automatic_mounting_data:
                print(f"    ⚠️ Пропускаю автомат {auto_number}: нет данных о монтаже")
                continue
                
            mounting_type, final_bbox = automatic_mounting_data[auto_box]
            
            # Получаем узел автомата из графа
            auto_node_id = node_ids.get(auto_box)
            if auto_node_id is None:
                print(f"    ⚠️ Пропускаю автомат {auto_number}: не найден узел в графе")
                continue
                
            auto_node = circuit_graph.nodes.get(auto_node_id)
            if auto_node is None:
                print(f"    ⚠️ Пропускаю автомат {auto_number}: узел не существует")
                continue

            # Создаем chunk с информацией (без текста, он будет распознан позже)
            x1, y1, x2, y2, conf, cls_id, cx, cy = text_box
            chunk_with_context = {
                "bbox": [x1, y1, x2, y2],  # Координаты текстового блока
                "confidence": float(conf),  # Уверенность детекции
                "class_id": int(cls_id),    # ID класса (обычно 8 для текста)
                "automatic_number": auto_number,  # Номер автомата
                "connected_element_type": 'automatic',  # Тип связанного элемента
                "connected_element_bbox": auto_node['bbox'],  # Bbox элемента
                "mounting_type": mounting_type,  # Тип монтажа автомата
                "automatic_final_bbox": final_bbox,  # Финальный bbox (с учетом монтажа)
                "text_number": text_number,  # Номер текста для отладки и визуализации
                "automatic_center": auto_node.get('center', (cx, cy)),  # Центр элемента
                "text_center": (cx, cy)  # Центр текстового блока
            }
            
            connected_text_chunks.append(chunk_with_context)
            
            print(f"    ✅ Автомат {auto_number} (монтаж {mounting_type}) → Текст {text_number}")
            print(f"      📍 Координаты текста: [{x1}, {y1}, {x2}, {y2}]")
            print(f"      📍 Координаты автомата: {auto_node['bbox']}")
            print(f"      📍 Финальный bbox (с монтажом): {final_bbox}")
        else:
            print(f"    ℹ️ Автомат {box_numbers.get(auto_box, '?')} не имеет связанного текста")

    # ШАГ 3: Создаем чанки для трансформаторов
    print("🔄 Создаю чанки для трансформаторов...")
    
    for edge in circuit_graph.edges:
        node1_id, node2_id, connection_type, distance = edge
        node1 = circuit_graph.nodes[node1_id]
        node2 = circuit_graph.nodes[node2_id]

        # Определяем связь "трансформатор-текст"
        if (node1['type'] == 'transformer' and node2['type'] == 'text'):
            transformer_node = node1
            text_node = node2
        elif (node1['type'] == 'text' and node2['type'] == 'transformer'):
            transformer_node = node2
            text_node = node1
        else:
            continue  # Пропускаем если это не связь трансформатор-текст

        # Проверяем, что этот текст еще не добавлен через связь с автоматом
        # Для этого ищем, есть ли уже чанк с таким text_node['id']
        already_added = False
        for chunk in connected_text_chunks:
            # Проверяем по координатам текстового блока
            chunk_bbox = chunk.get("bbox")
            if chunk_bbox:
                # Находим текстовый бокс, соответствующий этому узлу
                matching_text_box = None
                for text in text_boxes:
                    if text_nodes.get(text) == text_node['id']:
                        matching_text_box = text
                        break
                
                if matching_text_box and chunk_bbox == [matching_text_box[0], matching_text_box[1], matching_text_box[2], matching_text_box[3]]:
                    already_added = True
                    break
        
        if already_added:
            continue  # Этот текст уже добавлен через связь с автоматом

        # Находим соответствующий текстовый бокс
        text_box = None
        for text in text_boxes:
            if text_nodes.get(text) == text_node['id']:
                text_box = text
                break
        
        if text_box is None:
            print(f"    ⚠️ Пропускаю трансформатор: не найден текстовый бокс для узла {text_node['id']}")
            continue
        
        # Получаем номер текста
        text_number = text_numbers.get(text_box)
        if text_number is None:
            print(f"    ⚠️ Пропускаю трансформатор: не найден номер текста")
            continue

        # Получаем номер трансформатора из box_numbers
        transformer_number = None
        for box, number in box_numbers.items():
            if node_ids.get(box) == transformer_node['id']:
                transformer_number = number
                break
        
        if transformer_number is None:
            print(f"    ⚠️ Пропускаю трансформатор: не найден номер")
            continue

        # Создаем chunk для трансформатора
        x1, y1, x2, y2, conf, cls_id, cx, cy = text_box
        chunk_with_context = {
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "class_id": int(cls_id),
            "connected_element_type": 'transformer',
            "connected_element_bbox": transformer_node['bbox'],
            "transformer_number": transformer_number,  # Номер трансформатора
            "text_number": text_number,
            "text_center": (cx, cy),
            "transformer_center": transformer_node.get('center', (cx, cy))
        }
        
        connected_text_chunks.append(chunk_with_context)
        print(f"    ✅ Трансформатор {transformer_number} → Текст {text_number}")

    # ШАГ 4: Создаем чанки для УЗО
    print("🔄 Создаю чанки для УЗО...")
    
    for edge in circuit_graph.edges:
        node1_id, node2_id, connection_type, distance = edge
        node1 = circuit_graph.nodes[node1_id]
        node2 = circuit_graph.nodes[node2_id]

        # Определяем связь "УЗО-текст"
        if (node1['type'] == 'rcd' and node2['type'] == 'text'):
            rcd_node = node1
            text_node = node2
        elif (node1['type'] == 'text' and node2['type'] == 'rcd'):
            rcd_node = node2
            text_node = node1
        else:
            continue  # Пропускаем если это не связь УЗО-текст

        # Проверяем, что этот текст еще не добавлен
        already_added = False
        for chunk in connected_text_chunks:
            chunk_bbox = chunk.get("bbox")
            if chunk_bbox:
                # Находим текстовый бокс
                matching_text_box = None
                for text in text_boxes:
                    if text_nodes.get(text) == text_node['id']:
                        matching_text_box = text
                        break
                
                if matching_text_box and chunk_bbox == [matching_text_box[0], matching_text_box[1], matching_text_box[2], matching_text_box[3]]:
                    already_added = True
                    break
        
        if already_added:
            continue

        # Находим текстовый бокс
        text_box = None
        for text in text_boxes:
            if text_nodes.get(text) == text_node['id']:
                text_box = text
                break
        
        if text_box is None:
            continue
        
        # Получаем номер текста
        text_number = text_numbers.get(text_box)
        if text_number is None:
            continue

        # Получаем номер УЗО
        rcd_number = None
        for box, number in box_numbers.items():
            if node_ids.get(box) == rcd_node['id']:
                rcd_number = number
                break
        
        if rcd_number is None:
            continue

        # Создаем chunk для УЗО
        x1, y1, x2, y2, conf, cls_id, cx, cy = text_box
        chunk_with_context = {
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "class_id": int(cls_id),
            "connected_element_type": 'rcd',
            "connected_element_bbox": rcd_node['bbox'],
            "rcd_number": rcd_number,  # Номер УЗО
            "text_number": text_number,
            "text_center": (cx, cy),
            "rcd_center": rcd_node.get('center', (cx, cy))
        }
        
        connected_text_chunks.append(chunk_with_context)
        print(f"    ✅ УЗО {rcd_number} → Текст {text_number}")

    # ШАГ 5: Логируем итоговые результаты
    print(f"📊 ИТОГО: Создано {len(connected_text_chunks)} чанков для отправки в text-service")
    
    # Группируем по типам элементов для статистики
    automatic_chunks = [c for c in connected_text_chunks if c.get("connected_element_type") == 'automatic']
    transformer_chunks = [c for c in connected_text_chunks if c.get("connected_element_type") == 'transformer']
    rcd_chunks = [c for c in connected_text_chunks if c.get("connected_element_type") == 'rcd']
    
    print(f"  📋 Распределение по типам:")
    print(f"    • Автоматы: {len(automatic_chunks)} чанков")
    print(f"    • Трансформаторы: {len(transformer_chunks)} чанков")
    print(f"    • УЗО: {len(rcd_chunks)} чанков")
    
    # Выводим детальную информацию о чанках автоматов (для отладки)
    if automatic_chunks:
        print(f"  📋 Детали чанков автоматов:")
        for chunk in automatic_chunks:
            print(f"    • Автомат {chunk.get('automatic_number')}: "
                  f"Текст {chunk.get('text_number')}, "
                  f"Монтаж {chunk.get('mounting_type', '0')}, "
                  f"Bbox текста: {chunk.get('bbox')}")

    return connected_text_chunks

def visualize_connections(circuit_graph, image):
    for edge in circuit_graph.edges:
        node1_id, node2_id, connection_type, distance = edge
        node1 = circuit_graph.nodes[node1_id]
        node2 = circuit_graph.nodes[node2_id]

        # Определяем цвета для разных типов связей
        if 'automatic' in connection_type:
            color = (0, 255, 0)  # Зеленый для автоматов
        elif 'transformer' in connection_type:
            color = (255, 0, 0)  # Красный для трансформаторов
        elif 'rcd' in connection_type:
            color = (0, 0, 255)  # Синий для УЗО
        else:
            color = (255, 255, 0)  # Желтый для остального

        # Рисуем линию связи
        x1, y1 = int(node1['center'][0]), int(node1['center'][1])
        x2, y2 = int(node2['center'][0]), int(node2['center'][1])
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

        # Рисуем точку в центре элементов
        cv2.circle(image, (x1, y1), 4, color, -1)
        cv2.circle(image, (x2, y2), 4, color, -1)

def visualize_automatics_with_mounting(image, automatics_boxes, automatic_mounting_data, box_numbers, thickness, font_scale):
    """Визуализирует automatics с учетом типа монтажа"""
    print("  🎨 Визуализация автоматических выключателей...")
    
    for auto_box in automatics_boxes:
        mounting_type, final_bbox = automatic_mounting_data[auto_box]
        label = str(box_numbers[auto_box])
        color = colors[0 % len(colors)]  # cls_id = 0 для automatics
        
        # Используем финальный bbox (объединенный или оригинальный)
        x1, y1, x2, y2 = final_bbox
        
        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Добавляем номер
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
        
        print(f"    🎨 Автомат {label}: монтаж {mounting_type}, bbox {final_bbox}")

def visualize_other_elements(image, all_other_elements_sorted, box_numbers, thickness, font_scale):
    """Визуализирует остальные элементы"""
    print("  🎨 Визуализация остальных элементов...")
    
    for element in all_other_elements_sorted:
        x1, y1, x2, y2, conf, cls_id, cx, cy = element
        color = colors[cls_id % len(colors)]
        label = str(box_numbers[element])

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

def visualize_rectangles_for_texts(image, text_boxes_sorted, text_nodes, text_numbers, thickness):
    """
    Визуализирует текстовые блоки на изображении с номерами.
    
    Args:
        image: Изображение для рисования
        text_boxes_sorted: Отсортированный список текстовых bounding box
        text_nodes: Словарь сопоставления текстовых bbox → ID узлов в графе
        text_numbers: Словарь номеров текстовых блоков
        thickness: Толщина линий для рисования
    """
    print(f"🎨 Визуализация {len(text_boxes_sorted)} текстовых блоков...")
    
    # Цвет для текстовых блоков (класс 8)
    text_color = colors[8 % len(colors)]
    
    for text_box in text_boxes_sorted:
        try:
            x1, y1, x2, y2, conf, cls_id, cx, cy = text_box
            
            # Получаем номер текста из нумерации
            text_number = text_numbers.get(text_box)
            
            if text_number is None:
                print(f"  ⚠️ Пропускаю текстовый блок {text_box[:4]}: номер не найден")
                continue
            
            # ШАГ 1: Рисуем прямоугольник вокруг текстового блока
            cv2.rectangle(image, (x1, y1), (x2, y2), text_color, thickness)
            
            
            # Логируем для отладки
            print(f"  📝 Текст T{text_number}: bbox({x1}, {y1}, {x2}, {y2})")
            
        except Exception as e:
            print(f"  ❌ Ошибка при визуализации текстового блока: {e}")
            continue
    
    print(f"✅ Визуализация текстовых блоков завершена")

def send_to_yolo_service(image, yolo_url):
    """Отправка изображения в YOLO-сервис для детекции"""
    _, img_encoded = cv2.imencode('.png', image)
    files = {'file': ('image.png', img_encoded.tobytes(), 'image/png')}
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
        
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при отправке в YOLO-сервис: {e}")
        raise

def process_search_results(list_for_searching, list_for_quantity, class_counts, mounting_types_dict):
    """Обрабатывает результаты поиска и формирует запросы к БД"""
    request_results = []
    request_wh_results = []
    request_trans_results = []

    # Обработка автоматических выключателей
    quantity_qf = class_counts.get(0.0, 0)
    
    print(f"🔧 ОБРАБОТКА РЕЗУЛЬТАТОВ ПОИСКА:")
    print(f"  - Найдено автоматов в text-service: {len(list_for_searching)}")
    print(f"  - Ожидается по детекции: {quantity_qf}")
    print(f"  - Типы монтажа: {mounting_types_dict}")

    # ИСПРАВЛЕНИЕ: Используем МЕНЬШЕЕ значение из двух
    # all_qf_quantity = len(mounting_types_dict)
    actual_quantity = quantity_qf - len(list_for_searching)
    # min(quantity_qf, len(list_for_searching))
    print(f"  - Фактическое количество для обработки: {actual_quantity}")

    # Обрабатываем только нужное количество объектов
    
    if actual_quantity > 0:
        for i in range(actual_quantity):
            auto_number = i + 1  # Номер автомата соответствует порядку в списке
            qf_obj = list_for_searching[i] if i < len(list_for_searching) else None
            
            # Получаем mounting_type из словаря
            mounting_type = mounting_types_dict.get(auto_number, "0")
            list_for_searching.append(qf.create_qf('', '', '', '', '', '', mounting_type))
            
            # Если mounting_type не совпадает, обновляем
            # if qf_obj.Mounting_Type != mounting_type:
            #     qf_obj.Mounting_Type = mounting_type

    for qf_obj in list_for_searching:
        print(f"  🔧 Автомат {qf_obj}:")
        print(f"    ID: {repr(qf_obj.ID_QF)}")
        print(f"    Ток: {repr(qf_obj.Current)}")
        print(f"    Напряжение: {repr(qf_obj.Voltage)}")
        print(f"    Откл. способность: {repr(qf_obj.Current_Close)}")
        print(f"    Тип монтажа: {repr(qf_obj.Mounting_Type)}")
        
        request_results.append(db_client.search_breakers(qf_obj))

    # Обработка счетчиков (без изменений)
    if class_counts.get(2.0, 0) != 0:
        for i in range(class_counts.get(2.0, 0)):
            request_wh_results.append(db_client.search_counters())

    # Обработка трансформаторов 
    if class_counts.get(1.0, 0) != 0 or len(list_for_quantity) != 0:
        quantity_trans = class_counts.get(1.0, 0)
        if len(list_for_quantity) != 0:
            for ta_obj in list_for_quantity:
                quantity_trans += ta_obj.ta_quantity

        for i in range(quantity_trans):  
            request_trans_results.append(db_client.search_transformators())
    
    return request_results, request_wh_results, request_trans_results

def print_search_results(request_results, request_trans_results, request_wh_results):
    """Вывод результатов поиска в консоль"""
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
   yolo_local =  "http://localhost:5001"
   try:
        # Отправка в YOLO-сервис
        yolo_data = send_to_yolo_service(image, yolo_url)
        
        # Подготовка результатов для draw_rect
        results = [{
            'results': yolo_data['results'],
            'names': yolo_data['names']
        }]
        
        
        list_for_searching, list_for_quantity, mounting_types = draw_rect(results, image)
        
        # Подсчет классов
        class_ids = [int(box[5]) for box in yolo_data['results']]
        class_counts = Counter(class_ids)
        
        # Обработка результатов поиска
        request_results, request_wh_results, request_trans_results = process_search_results(
            list_for_searching, list_for_quantity, class_counts, mounting_types
        )

        # Вывод результатов
        print_search_results(request_results, request_trans_results, request_wh_results)

        # Формирование итоговых результатов
        results_of_searching = [
            request_results,
            request_trans_results,
            request_wh_results
        ]
        
        results_list = [
            image,
            results_of_searching
        ]
        
        return results_list

   except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise



if __name__ == "__main__":
    # model = YOLO("./runs/restudying_neuro_v5.71s/weights/best.pt") 
    result = detect_one_image("C:/nekitlox/NeuroForCircuit/tests/RU.png")