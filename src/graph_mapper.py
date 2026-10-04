
def create_graphs(circuit_graph, automatics_boxes, transformer_boxes, rcd_boxes, text_boxes):
    node_ids = {}
    text_nodes = {}

    def add_boxes_to_graph(boxes, node_type, target_dict):
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id, cx, cy = box
            
            node_id = circuit_graph.add_node(
                bbox=[x1, y1, x2, y2],
                node_type=node_type,
                confidence=conf
            )
            
            if node_id in circuit_graph.nodes:
                circuit_graph.nodes[node_id]['type'] = node_type

            
            target_dict[tuple(box)] = node_id

    # Наполнение графа узлами
    add_boxes_to_graph(automatics_boxes, 'automatic', node_ids)
    add_boxes_to_graph(transformer_boxes, 'transformer', node_ids)
    add_boxes_to_graph(rcd_boxes, 'rcd', node_ids)
    add_boxes_to_graph(text_boxes, 'text', text_nodes)

    print("Создание АДАПТИВНЫХ связей в графе...")

    for node_type in ['transformer', 'automatic', 'rcd']:
        circuit_graph.create_exclusive_connections_adaptive(node_type)

    print(f"Граф построен: {len(circuit_graph.nodes)} узлов, {len(circuit_graph.edges)} связей")
    return node_ids, text_nodes


def create_a_mind_map(
    circuit_graph, 
    text_boxes, 
    automatics_boxes, 
    node_ids, 
    text_nodes, 
    transformer_boxes=None, 
    rcd_boxes=None):

    # Разделение карты связей по типам элементов
    maps = {
        'automatic': {'text_to_elem': {}, 'elem_to_text': {}},
        'transformer': {'text_to_elem': {}, 'elem_to_text': {}},
        'rcd': {'text_to_elem': {}, 'elem_to_text': {}}
    }

    transformer_boxes = transformer_boxes or []
    rcd_boxes = rcd_boxes or []

    # Создание отдельных справочников поиска для каждого типа
    box_lookup = {
        'automatic': automatics_boxes,
        'transformer': transformer_boxes,
        'rcd': rcd_boxes
    }

    # Множество для отслеживания уже привязанных текстовых узлов
    used_text_nodes = set()

    for edge in circuit_graph.edges:
        node1_id, node2_id = edge[0], edge[1]
        
        node1 = circuit_graph.nodes.get(node1_id)
        node2 = circuit_graph.nodes.get(node2_id)

        if not node1 or not node2:
            continue

        n1_type = node1.get('type') or node1.get('node_type')
        n2_type = node2.get('type') or node2.get('node_type')

        # Определение типа целевого элемента
        target_type = None
        if n1_type == 'text' and n2_type in ['automatic', 'transformer', 'rcd']:
            text_node_id, elem_node_id = node1_id, node2_id
            target_type = n2_type
        elif n2_type == 'text' and n1_type in ['automatic', 'transformer', 'rcd']:
            text_node_id, elem_node_id = node2_id, node1_id
            target_type = n1_type
        else:
            continue

        # Если этот текстовый узел уже привязан к объекту с более высоким приоритетом - пропуск
        if text_node_id in used_text_nodes:
            continue

        text_box = None
        elem_box = None

        # Поиск текстового бокса
        for text in text_boxes:
            if text_nodes.get(tuple(text)) == text_node_id:
                text_box = text
                break

        # Поиск бокса элемента строго внутри его типа
        for elem in box_lookup[target_type]:
            if node_ids.get(tuple(elem)) == elem_node_id:
                elem_box = elem
                break

        if text_box and elem_box:
            maps[target_type]['text_to_elem'][tuple(text_box)] = elem_box
            maps[target_type]['elem_to_text'][tuple(elem_box)] = text_box
            used_text_nodes.add(text_node_id)

    # Для обратной совместимости возвращаем агрегированные словари, 
    # но также отдаем разделенные по типам карты
    text_to_all_map = {}
    all_to_text_map = {}
    for t in maps:
        text_to_all_map.update(maps[t]['text_to_elem'])
        all_to_text_map.update(maps[t]['elem_to_text'])

    return text_to_all_map, all_to_text_map, maps

def create_connect_text_chunks(circuit_graph, automatics_with_texts, automatic_to_text_map, 
                              box_numbers, node_ids, text_nodes, text_boxes, 
                              text_numbers, automatic_mounting_data):
    """
    Создает связанные текстовые чанки с использованием уже имеющейся нумерации.
    
    Эта функция анализирует связи между элементами (автоматами, трансформаторами, УЗО) 
    и текстовыми блоками в графе, создавая структурированные чанки для последующей
    обработки text-service. Текст НЕ распознается в этой функции, а добавляется позже.
    
    Args:
        - circuit_graph: Граф с элементами и связями
        - automatics_with_texts: Список автоматов, связанных с текстами
        - automatic_to_text_map: Словарь сопоставления автомат → текст
        - box_numbers: Словарь номеров элементов (автоматов, трансформаторов и т.д.)
        - node_ids: Словарь сопоставления bounding box → ID узла в графе
        - text_nodes: Словарь сопоставления текстового bbox → ID текстового узла
        - text_boxes: Список всех текстовых bounding box
        - text_numbers: Словарь нумерации текстовых блоков (уже отсортированных)
        - automatic_mounting_data: Данные о типе монтажа для автоматов
    
    Returns:
        List[dict]: Список чанков со структурой для отправки в text-service
    """
    connected_text_chunks = []

    def find_text_box_by_node(text_node_id):
        for text in text_boxes:
            if text_nodes.get(text) == text_node_id:
                return text
        return None
    
    def is_text_already_added(matching_text_box):
        if not matching_text_box:
            return False
        target_bbox = list(matching_text_box[:4])
        for chunk in connected_text_chunks:
            if chunk.get("bbox") == target_bbox:
                return True
        return False

    # ШАГ 1: Сбор всех текстовые узлов, связанных с любыми элементами

    print("Анализ связи в графе для фильтрации текстов...")
    connected_text_nodes = set()
    
    for edge in circuit_graph.edges:
        node1_id, node2_id = edge[:2]
        node1 = circuit_graph.nodes[node1_id]
        node2 = circuit_graph.nodes[node2_id]

        # Определяем связи "элемент-текст"
        # Вариант 1: узел1 - элемент, узел2 - текст
        if (node1['type'] != 'text' and node2['type'] == 'text'):
            connected_text_nodes.add(node2['id'])
            
        # Вариант 2: узел1 - текст, узел2 - элемент
        elif (node1['type'] == 'text' and node2['type'] != 'text'):
            connected_text_nodes.add(node1['id'])
    
    print(f"Найдено {len(connected_text_nodes)} текстовых узлов, связанных с элементами")

    # ШАГ 2: Создаем чанки для автоматов
    print("Создание чанков для автоматов...")
    
    for auto_box in automatics_with_texts:
        auto_key = tuple(auto_box)
        if auto_key not in automatic_to_text_map:
            print(f"Автомат {box_numbers.get(auto_box, '?')} не имеет связанного текста")
            continue
            
        text_box = automatic_to_text_map[auto_key]
        text_number = text_numbers.get(tuple(text_box))
        auto_number = box_numbers.get(auto_box)
        
        if text_number is None or auto_number is None or auto_box not in automatic_mounting_data:
            print(f"Пропуск автомата: отсутствуют необходимые данные или индексы")
            continue
            
        mounting_type, final_bbox = automatic_mounting_data[auto_box]
        auto_node_id = node_ids.get(auto_box)
        if auto_node_id is None or auto_node_id not in circuit_graph.nodes:
            print(f"Пропуск автомата: {auto_number}: узел не найден в графе")
            continue
            
        auto_node = circuit_graph.nodes[auto_node_id]
        x1, y1, x2, y2, conf, cls_id, cx, cy = text_box
        
        chunk_with_context = {
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "class_id": int(cls_id),
            "automatic_number": auto_number,
            "connected_element_type": 'automatic',
            "connected_element_bbox": auto_node['bbox'],
            "mounting_type": mounting_type,
            "automatic_final_bbox": final_bbox,
            "text_number": text_number,
            "automatic_center": auto_node.get('center', (cx, cy)),
            "text_center": (cx, cy)
        }
        connected_text_chunks.append(chunk_with_context)
        print(f"Автомат {auto_number} (монтаж {mounting_type}) -> Текст {text_number}")


    # ШАГ 3: Создание чанка для элемента
    def process_edge_elements(target_type, element_label_name):
        print(f"Создание чанков для {element_label_name}...")
        for edge in circuit_graph.edges:
            # Распаковка ребра
            node1_id, node2_id = edge[:2]
            
            node1 = circuit_graph.nodes.get(node1_id)
            node2 = circuit_graph.nodes.get(node2_id)
            
            if not node1 or not node2:
                continue

            n1_type = node1.get('type') or node1.get('node_type')
            n2_type = node2.get('type') or node2.get('node_type')

            # Определение ориентации связи (элемент <-> текст)
            if n1_type == target_type and n2_type == 'text':
                elem_node, text_node = node1, node2
            elif n1_type == 'text' and n2_type == target_type:
                elem_node, text_node = node2, node1
            else:
                continue

            # Безопасное получение ID текстового узла
            text_node_id = text_node.get('id') or text_node_id
            text_box = find_text_box_by_node(text_node_id)
            
            if text_box is None or is_text_already_added(text_box):
                continue

            text_number = text_numbers.get(text_box)
            if text_number is None:
                continue

            # Поиск номера элемента по его боксу в графе
            elem_number = None
            for box, number in box_numbers.items():
                if node_ids.get(box) == elem_node.get('id'):
                    elem_number = number
                    break
            
            if elem_number is None:
                continue

            x1, y1, x2, y2, conf, cls_id, cx, cy = text_box
            chunk_with_context = {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls_id),
                "connected_element_type": target_type,
                "connected_element_bbox": elem_node.get('bbox', [0,0,0,0]),
                f"{target_type}_number": elem_number,
                "text_number": text_number,
                "text_center": (cx, cy),
                f"{target_type}_center": elem_node.get('center', (cx, cy))
            }
            
            connected_text_chunks.append(chunk_with_context)
            print(f"{element_label_name} {elem_number} -> Текст {text_number}")

    # ШАГ 4: Обрабатка трансформаторов и УЗО с помощью общей функции
    process_edge_elements('transformer', 'Трансформатор')
    process_edge_elements('rcd', 'УЗО')

    # ШАГ 5: Итоговая статистика
    print(f"ИТОГО: Создано {len(connected_text_chunks)} чанков для отправки в text-service")
    return connected_text_chunks