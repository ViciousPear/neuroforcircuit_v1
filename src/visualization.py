import cv2

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (192, 192, 192), (128, 128, 128), (128, 128, 0),
    (128, 0, 0), (0, 128, 0), (128, 0, 128),
    (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209),
    (148, 0, 211), (255, 20, 147)
]

def draw_labeled_box(image, box, label, color, thickness=2, font_scale=0.8):
    """
    Визуализирует bounding box на изображении с номером (меткой).

    Рисует прямоугольник по координатам box, а над верхней гранью —
    залитую плашку с номером элемента и сам номер белым цветом.

    Args:
        image (np.ndarray): Изображение (OpenCV BGR), на котором
            выполняется отрисовка. Изменяется in-place.
        box (tuple | list): Bounding box в формате [x1, y1, x2, y2, ...];
            используются первые 4 элемента.
        label (str): Текст метки (обычно номер элемента).
        color (tuple[int, int, int]): Цвет прямоугольника и плашки (BGR).
        thickness (int): Толщина линий. По умолчанию 2.
        font_scale (float): Масштаб шрифта. По умолчанию 0.8.

    Returns:
        None: Изображение модифицируется in-place.
    """
    x1, y1, x2, y2 = box[:4]

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


def render_circuit_visualization(
        image,
        automatics_boxes,
        automatic_mounting_data,
        all_other_elements_sorted,
        text_boxes_sorted,
        text_numbers,
        box_numbers,
        circuit_graph=None,
        thickness=2,
        font_scale=0.8
):
    """
    Визуализирует все элементы схемы в зависимости от их типа.

    Порядок отрисовки:
        1. Автоматические выключатели (automatics_boxes) — с учетом
           типа монтажа: используется final_bbox из
           automatic_mounting_data (объединенный bbox для типа "1").
        2. Остальные элементы (all_other_elements_sorted) — трансформаторы,
           УЗО, счетчики и прочее; цвет выбирается по cls_id.
        3. Текстовые блоки (text_boxes_sorted) — обводятся рамкой,
           цвет фиксированный (colors[8]).
        4. Если передан circuit_graph — рисуются связи (ребра) между
           узлами графа: линии между центрами и точки в центрах.
           Цвет ребра зависит от типа связи ('automatic' → зеленый,
           'transformer' → красный, 'rcd' → синий, иначе — желтый).

    Args:
        image (np.ndarray): Изображение (OpenCV BGR), на котором
            выполняется отрисовка. Изменяется in-place.
        automatics_boxes (list): Bbox автоматических выключателей
            (используются как ключи для automatic_mounting_data и
            box_numbers).
        automatic_mounting_data (dict): Карта
            {auto_bbox: (mounting_type, final_bbox)}, где final_bbox —
            bbox для отрисовки (объединенный при типе монтажа "1").
        all_other_elements_sorted (list): Отсортированные bbox
            остальных элементов в формате
            (x1, y1, x2, y2, conf, cls_id, cx, cy).
        text_boxes_sorted (list): Отсортированные bbox текстовых
            блоков.
        text_numbers (dict): Карта {text_bbox: номер_текста}.
        box_numbers (dict): Карта {bbox: номер_элемента} для всех
            элементов схемы.
        circuit_graph (CircuitGraph | None): Граф связей. Если None —
            связи не рисуются. По умолчанию None.
        thickness (int): Толщина линий. По умолчанию 2.
        font_scale (float): Масштаб шрифта. По умолчанию 0.8.

    Returns:
        None: Изображение модифицируется in-place.

    Notes:
        - Ошибки при визуализации отдельного текстового блока
          логируются и не прерывают отрисовку остальных элементов.
        - Параметр circuit_graph предназначен в первую очередь для
          отладки/тестирования; в production-пайплайне передается None.
    """
    for auto_box in automatics_boxes:
        mounting_type, final_bbox = automatic_mounting_data[auto_box]
        label = str(box_numbers[auto_box])
        color = colors[0 % len(colors)]

        draw_labeled_box(image, final_bbox, label, color, thickness, font_scale)

    for element in all_other_elements_sorted:
        x1, y1, x2, y2, conf, cls_id, cx, cy = element
        color = colors[cls_id % len(colors)]
        label = str(box_numbers[element])

        draw_labeled_box(image, element, label, color, thickness, font_scale)

    text_color = colors[8 % len(colors)]
    for text_box in text_boxes_sorted:
        try:
            if text_box in text_numbers:
                x1, y1, x2, y2 = text_box[:4]
                cv2.rectangle(image, (x1, y1), (x2, y2), text_color, thickness)

        except Exception as e:
            print(f"  Ошибка при визуализации текстового блока: {e}")
            continue

    if circuit_graph is not None:
        for edge in circuit_graph.edges:
            node1_id, node2_id, connection_type, distance = edge
            node1 = circuit_graph.nodes[node1_id]
            node2 = circuit_graph.nodes[node2_id]

            # Цвета для разных типов связи
            if 'automatic' in connection_type:
                # Зеленый для автоматов
                color = (0, 255, 0)
            elif 'transformer' in connection_type:
                # Красный для трансформаторов
                color = (255, 0, 0)
            elif 'rcd' in connection_type:
                # Синий для УЗО
                color = (0, 0, 255)
            else:
                # Желтый для остального
                color = (255, 255, 0)

            # Линия связи
            x1, y1 = int(node1['center'][0]), int(node1['center'][1])
            x2, y2 = int(node2['center'][0]), int(node2['center'][1])
            cv2.line(image, (x1, y1), (x2, y2), color, 2)

            # Точка в центре элемента
            cv2.circle(image, (x1, y1), 4, color, -1)
            cv2.circle(image, (x2, y2), 4, color, -1)