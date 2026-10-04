def determine_mounting_type_and_bbox(automatic_box, bracing_boxes, max_distance=40):
    """
    Определяет тип монтажа автомата и, при необходимости, возвращает
    объединенный bounding box.

    Логика:
        - Ищет bracing-объекты сверху и снизу от автомата (см. find_nearby_bracing).
        - Если найден хотя бы один bracing сверху И хотя бы один снизу —
          тип монтажа "1", а bbox объединяется по всем связанным объектам
          (автомат + все найденные bracing).
        - Иначе — тип монтажа "0", возвращается исходный bbox автомата.

    Args:
        automatic_box (tuple | list): Данные автомата в формате
            (x1, y1, x2, y2, conf, cls_id, cx, cy).
        bracing_boxes (list): Список bracing-объектов в том же формате.
        max_distance (int | float): Максимальное вертикальное расстояние
            (в пикселях) для поиска bracing. По умолчанию 40.

    Returns:
        tuple[str, tuple[float, float, float, float]]:
            - Тип монтажа: "1" или "0".
            - Bounding box: объединенный (для типа "1") либо bbox автомата
              (для типа "0") в формате (min_x, min_y, max_x, max_y).
    """
    x1_a, y1_a, x2_a, y2_a, conf_a, cls_id_a, cx_a, cy_a = automatic_box

    print(f"Анализ автомата: bbox({x1_a}, {y1_a}, {x2_a}, {y2_a})")
    print(f"Всего bracing объектов: {len(bracing_boxes)}")

    top_bracings, bottom_bracings = find_nearby_bracing(automatic_box, bracing_boxes, max_distance)

    # Если есть хотя бы один bracing сверху и хотя бы один снизу
    if len(top_bracings) >= 1 and len(bottom_bracings) >= 1:
        # Поиск крайних точек среди всех связанных bracing и automatic
        all_related_boxes = [automatic_box] + top_bracings + bottom_bracings

        min_x = min(box[0] for box in all_related_boxes)
        min_y = min(box[1] for box in all_related_boxes)
        max_x = max(box[2] for box in all_related_boxes)
        max_y = max(box[3] for box in all_related_boxes)

        print(f"Монтаж тип 1: объединенный bbox({min_x}, {min_y}, {max_x}, {max_y})")
        return "1", (min_x, min_y, max_x, max_y)
    else:
        print(f"Монтаж тип 0: недостаточно bracing (сверху: {len(top_bracings)}, снизу: {len(bottom_bracings)})")
        return "0", automatic_box[:4]


def find_nearby_bracing(automatic_box, bracing_boxes, max_distance=45):
    """
    Находит bracing-объекты сверху и снизу от автомата в пределах max_distance.

    Критерии отбора bracing:
        - Горизонтальное перекрытие с автоматом > 0.3 (см. calculate_horizontal_overlap).
        - Вертикальное расстояние от границы автомата до ближайшей границы
          bracing > 0 и <= max_distance (см. calculate_vertical_distance_from_edges).

    Объекты классифицируются как top (выше автомата) или bottom (ниже автомата)
    в зависимости от взаимного расположения по вертикали.

    Args:
        automatic_box (tuple | list): Данные автомата в формате
            (x1, y1, x2, y2, conf, cls_id, cx, cy).
        bracing_boxes (list): Список bracing-объектов в том же формате.
        max_distance (int | float): Максимальное вертикальное расстояние
            (в пикселях). По умолчанию 45.

    Returns:
        tuple[list, list]:
            - top_bracings: список bracing выше автомата.
            - bottom_bracings: список bracing ниже автомата.
    """
    x1_a, y1_a, x2_a, y2_a, conf_a, cls_id_a, cx_a, cy_a = automatic_box

    top_bracings = []
    bottom_bracings = []

    print(f"Поиск bracing для автомата: Y=[{y1_a}-{y2_a}], высота={y2_a - y1_a}px")

    for i, bracing in enumerate(bracing_boxes):
        x1_b, y1_b, x2_b, y2_b, conf_b, cls_id_b, cx_b, cy_b = bracing

        print(f"Анализ bracing {i + 1}: Y=[{y1_b}-{y2_b}], высота={y2_b - y1_b}px")

        # Проверка горизонтального перекрытия (должно быть значительным)
        horizontal_overlap = calculate_horizontal_overlap(automatic_box, bracing)

        if horizontal_overlap > 0.3:
            # Вычисление расстояния от границ и определяем положение
            vertical_distance, is_above = calculate_vertical_distance_from_edges(automatic_box, bracing)

            if vertical_distance <= max_distance and vertical_distance > 0:
                if is_above:
                    # bracing выше automatic
                    top_bracings.append(bracing)
                    print(f"ДОБАВЛЕНО СВЕРХУ: расстояние={vertical_distance:.1f}px")
                else:
                    # bracing ниже automatic
                    bottom_bracings.append(bracing)
                    print(f"ДОБАВЛЕНО СНИЗУ: расстояние={vertical_distance:.1f}px")
            else:
                print(f"ПРОПУЩЕНО: расстояние {vertical_distance:.1f}px > {max_distance}px")
        else:
            print(f"ПРОПУЩЕНО: слабое горизонтальное перекрытие {horizontal_overlap:.2f}")

    print(f"Итог: {len(top_bracings)} сверху, {len(bottom_bracings)} снизу")
    return top_bracings, bottom_bracings


def calculate_vertical_distance_from_edges(automatic_box, bracing_box):
    """
    Вычисляет вертикальное расстояние от верхней/нижней границы автомата
    до ближайшей границы bracing и определяет положение bracing.

    Логика:
        - Если bracing полностью выше автомата (y2_b < y1_a):
          расстояние = y1_a - y2_b, is_above = True.
        - Если bracing полностью ниже автомата (y1_b > y2_a):
          расстояние = y1_b - y2_a, is_above = False.
        - Если есть перекрытие по Y: расстояние = 0, is_above = False.

    Args:
        automatic_box (tuple | list): Данные автомата; используются первые
            4 элемента (x1, y1, x2, y2).
        bracing_box (tuple | list): Данные bracing; используются первые
            4 элемента (x1, y1, x2, y2).

    Returns:
        tuple[float, bool]:
            - vertical_distance: расстояние в пикселях (0 при перекрытии по Y).
            - is_above: True, если bracing выше автомата, иначе False.
    """
    x1_a, y1_a, x2_a, y2_a = automatic_box[:4]
    x1_b, y1_b, x2_b, y2_b = bracing_box[:4]

    # Расстояние от верхней границы automatic до нижней границы bracing (если bracing выше)
    distance_from_top = y1_a - y2_b if y2_b < y1_a else 0

    # Расстояние от нижней границы automatic до верхней границы bracing (если bracing ниже)
    distance_from_bottom = y1_b - y2_a if y1_b > y2_a else 0

    # Определение положения bracing относительно automatics
    # bracing полностью выше automatics
    is_above = y2_b < y1_a
    # bracing полностью ниже automatics
    is_below = y1_b > y2_a

    if is_above:
        vertical_distance = distance_from_top
        print(f"Bracing ВЫШЕ: расстояние от верха automatics до низа bracing = {vertical_distance:.1f}")
    elif is_below:
        vertical_distance = distance_from_bottom
        print(f"Bracing НИЖЕ: расстояние от низа automatics до верха bracing = {vertical_distance:.1f}")
    else:
        vertical_distance = 0  # есть перекрытие по Y
        print(f"Bracing ПЕРЕКРЫВАЕТСЯ по вертикали")

    return vertical_distance, is_above


def calculate_horizontal_overlap(box1, box2):
    """
    Вычисляет долю горизонтального перекрытия между двумя bounding box.

    Перекрытие нормируется на минимальную из двух ширин:
        overlap_percentage = overlap_x / min(width1, width2)

    Это позволяет корректно оценивать перекрытие даже для объектов
    сильно разного размера (например, автомат vs. короткий bracing).

    Args:
        box1 (tuple | list): Первый bounding box; используются первые
            4 элемента (x1, y1, x2, y2).
        box2 (tuple | list): Второй bounding box; используются первые
            4 элемента (x1, y1, x2, y2).

    Returns:
        float: Доля горизонтального перекрытия (0..1). Если минимальная
            ширина равна 0, возвращает 0.
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # Горизонтальное перекрытие
    overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    width1 = x2_1 - x1_1
    width2 = x2_2 - x1_2

    # Процент перекрытия по ширине
    overlap_percentage = overlap_x / min(width1, width2) if min(width1, width2) > 0 else 0
    return overlap_percentage