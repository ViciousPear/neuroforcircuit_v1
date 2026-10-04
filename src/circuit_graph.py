import math


class CircuitGraph:
    """
    Граф схемы: узлы (элементы и текстовые блоки) и связи между ними.

    Узел (node) — словарь с полями:
        - id: int — уникальный идентификатор
        - bbox: list[float] — bounding box [x1, y1, x2, y2]
        - center: tuple[float, float] — центр bbox
        - type: str — тип узла ('text', 'transformer', 'automatic' и т.п.)
        - confidence: float — уверенность 
        - text: str | None — текстовое содержимое (для текстовых узлов)
        - connections: list[tuple[int, str, float]] — связи (node_id, тип_связи, расстояние)
        - is_connected: bool — флаг, устанавливается при создании связи

    Ребро (edge) — кортеж (node1_id, node2_id, connection_type, distance).
    """

    def __init__(self, distance_multiplier=1.7):
        """
        Инициализирует пустой граф схемы.

        Args:
            distance_multiplier (float): Базовый множитель для расчета адаптивной
                максимальной дистанции между элементом и текстом. По умолчанию 1.7.

        Returns:
            None
        """
        self.nodes = {}
        self.edges = []
        self.next_id = 0
        self.use_orientation = False
        self.distance_multiplier = distance_multiplier

    def calculate_element_size(self, bbox):
        """
        Вычисляет ширину и высоту элемента по его bounding box.

        Args:
            bbox (list[float]): Bounding box в формате [x1, y1, x2, y2].

        Returns:
            tuple[float, float]: Кортеж (width, height) — ширина и высота элемента.
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height

    def calculate_adaptive_max_distance(self, bbox):
        """
        Вычисляет адаптивную максимальную дистанцию для поиска текста рядом
        с элементом, исходя из его размеров.

        Логика:
            - Берется максимальная сторона bbox.
            - Умножается на self.distance_multiplier.
            - Результат ограничивается сверху значением min(250.0, max_dim * 2.0)
              и снизу — 0.

        Args:
            bbox (list[float]): Bounding box элемента [x1, y1, x2, y2].

        Returns:
            float: Адаптивная максимальная дистанция в пикселях.
        """
        width, height = self.calculate_element_size(bbox)
        max_dimension = max(width, height)
        adaptive_distance = max_dimension * self.distance_multiplier

        min_distance = 0
        max_distance = min(250.0, max_dimension * 2.0)
        adaptive_distance = max(min_distance, min(adaptive_distance, max_distance))

        print(
            f"Adaptive Distance: size={width:.0f}x{height:.0f}, "
            f"max_dim={max_dimension:.0f}px, "
            f"multiplier={self.distance_multiplier}, "
            f"distance={adaptive_distance:.1f}px"
        )
        return adaptive_distance

    def calculate_adaptive_max_distance_special(self, bbox, element_type, multiplier):
        """
        Вычисляет адаптивную максимальную дистанцию с учетом типа элемента.

        Для трансформаторов применяется уменьшенный коэффициент (multiplier * 0.3)
        и более жесткие границы [10.0, 70.0], чтобы избежать ложных связей.
        Для остальных типов — стандартный коэффициент multiplier и границы [0.0, 250.0].

        Args:
            bbox (list[float]): Bounding box элемента [x1, y1, x2, y2].
            element_type (str): Тип элемента ('transformer', 'automatic' и т.п.).
            multiplier (float): Базовый множитель дистанции.

        Returns:
            float: Адаптивная максимальная дистанция в пикселях.
        """
        width, height = self.calculate_element_size(bbox)
        max_dimension = max(width, height)

        if element_type == 'transformer':
            adaptive_distance = max_dimension * (multiplier * 0.3)
            min_distance = 10.0
            max_distance = 70.0
        else:
            adaptive_distance = max_dimension * multiplier
            min_distance = 0.0
            max_distance = 250.0

        adaptive_distance = max(min_distance, min(adaptive_distance, max_distance))

        print(
            f"Adaptive Distance for {element_type}: "
            f"size={width:.0f}x{height:.0f}, "
            f"distance={adaptive_distance:.1f}px"
        )
        return adaptive_distance

    def check_connection_criteria(self, elem_bbox, text_bbox, distance, max_distance, vertical_overlap):
        """
        Проверяет общие критерии допустимости связи между элементом и текстовым узлом.

        Критерии (любой из них достаточен):
            - вертикальное перекрытие > 0.05;
            - направление преимущественно 'top'/'bottom' и горизонтальное перекрытие > 0.1;
            - расстояние меньше 30% от максимально допустимого.

        Args:
            elem_bbox (list[float]): Bounding box элемента.
            text_bbox (list[float]): Bounding box текстового узла.
            distance (float): Расстояние между границами bbox.
            max_distance (float): Максимально допустимая дистанция.
            vertical_overlap (float): Доля вертикального перекрытия (0..1).

        Returns:
            bool: True, если связь допустима, иначе False.
        """
        if vertical_overlap > 0.05:
            return True

        direction_info = self.get_direction_zone(elem_bbox, text_bbox)
        if direction_info['primary'] in ['top', 'bottom']:
            horizontal_overlap = self.get_horizontal_overlap(elem_bbox, text_bbox)
            if horizontal_overlap > 0.1:
                return True

        if distance < max_distance * 0.3:
            return True

        return False

    def create_exclusive_connections_adaptive(self, element_type):
        """
        Создает эксклюзивные адаптивные связи между элементами заданного типа
        и свободными текстовыми узлами.

        Каждый элемент получает не более одного текста, каждый текст — не более
        одного элемента. Выбор лучшей пары происходит по максимальному скору
        (calculate_connection_score_simple). Для трансформаторов критерий мягче:
        допускается любое ненулевое перекрытие или попадание в адаптивную дистанцию.

        Args:
            element_type (str): Тип элементов, для которых создаются связи
                ('automatic', 'transformer', 'rcd' и т.п.).

        Returns:
            int: Количество созданных связей.
        """
        print(
            f"Creating ADAPTIVE connections for {element_type} "
            f"(distance multiplier: {self.distance_multiplier})..."
        )

        # Сбор только элементов текущего типа
        element_nodes = [
            node_id for node_id, node_data in self.nodes.items()
            if (node_data.get('type') or node_data.get('node_type')) == element_type
        ]

        distance_multiplier_special = self.distance_multiplier

        # Сортировка элементов по Y, затем по X
        element_nodes.sort(key=lambda nid: (self.nodes[nid]['bbox'][1], self.nodes[nid]['bbox'][0]))

        connections_created = 0

        for element_id in element_nodes:
            element_data = self.nodes[element_id]

            # Динамическое получение всех свободных текстов прямо на момент обработки элемента
            free_text_nodes = [
                node_id for node_id, node_data in self.nodes.items()
                if (node_data.get('type') or node_data.get('node_type')) == 'text'
                and not node_data.get('is_connected', False)
            ]

            adaptive_max_distance = self.calculate_adaptive_max_distance_special(
                element_data['bbox'], element_type, distance_multiplier_special
            )

            possible_pairs = []

            for text_id in free_text_nodes:
                text_data = self.nodes[text_id]
                distance = self.calculate_distance(element_id, text_id)

                if distance <= adaptive_max_distance:
                    vertical_overlap = self.get_vertical_overlap(element_data['bbox'], text_data['bbox'])
                    horizontal_overlap = self.get_horizontal_overlap(element_data['bbox'], text_data['bbox'])
                    direction_info = self.get_direction_zone(element_data['bbox'], text_data['bbox'])

                    allow_pair = False

                    if element_type == 'transformer':
                        if (
                            distance <= adaptive_max_distance
                            or vertical_overlap > 0.0
                            or horizontal_overlap > 0.0
                        ):
                            allow_pair = True
                    else:
                        allow_pair = self.check_connection_criteria(
                            element_data['bbox'], text_data['bbox'],
                            distance, adaptive_max_distance, vertical_overlap
                        )

                    if allow_pair:
                        score = self.calculate_connection_score_simple(
                            element_id, text_id, adaptive_max_distance,
                            direction_info, vertical_overlap
                        )
                        possible_pairs.append({
                            'text_id': text_id,
                            'distance': distance,
                            'score': score,
                            'text_content': text_data.get('text', 'Unknown'),
                            'vertical_overlap': vertical_overlap,
                            'direction_info': direction_info
                        })

            if possible_pairs:
                possible_pairs.sort(key=lambda x: x['score'], reverse=True)
                best_pair = possible_pairs[0]

                self.add_edge(element_id, best_pair['text_id'], 'exclusive_adaptive')

                self.nodes[best_pair['text_id']]['is_connected'] = True
                self.nodes[element_id]['is_connected'] = True

                connections_created += 1

                direction_str = self.get_direction_string(best_pair['direction_info'])
                width, height = self.calculate_element_size(element_data['bbox'])

                print(
                    f"{element_type} (size: {width:.0f}x{height:.0f}px) → "
                    f"'{best_pair['text_content']}' "
                    f"(dist: {best_pair['distance']:.1f}px / {adaptive_max_distance:.1f}px, "
                    f"pos: {direction_str}, overlap: {best_pair['vertical_overlap']:.2f})"
                )
            else:
                width, height = self.calculate_element_size(element_data['bbox'])
                print(
                    f"{element_type} (size: {width:.0f}x{height:.0f}px) - no text found "
                    f"(max dist: {adaptive_max_distance:.1f}px)"
                )

        print(f"Created {connections_created} adaptive connections for {element_type}")
        return connections_created

    def lines_intersect(self, p1, p2, p3, p4):
        """
        Проверяет пересечение двух отрезков p1-p2 и p3-p4.

        Использует тест ориентации (counter-clockwise) для определения,
        пересекаются ли отрезки.

        Args:
            p1 (tuple[float, float]): Начало первого отрезка.
            p2 (tuple[float, float]): Конец первого отрезка.
            p3 (tuple[float, float]): Начало второго отрезка.
            p4 (tuple[float, float]): Конец второго отрезка.

        Returns:
            bool: True, если отрезки пересекаются, иначе False.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def add_node(self, bbox, node_type, confidence, text_content=None):
        """
        Добавляет новый узел в граф и возвращает его идентификатор.

        Args:
            bbox (list[float]): Bounding box узла [x1, y1, x2, y2].
            node_type (str): Тип узла ('text', 'automatic', 'transformer' и т.п.).
            confidence (float): Уверенность детекции.
            text_content (str | None): Текстовое содержимое (для текстовых узлов).

        Returns:
            int: Идентификатор созданного узла.
        """
        node_id = self.next_id
        self.next_id += 1

        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        self.nodes[node_id] = {
            'id': node_id,
            'bbox': bbox,
            'center': (center_x, center_y),
            'type': node_type,
            'confidence': confidence,
            'text': text_content,
            'connections': []
        }
        return node_id

    def calculate_bbox_distance(self, bbox1, bbox2):
        """
        Вычисляет кратчайшее расстояние между границами двух bounding box.

        Если bbox пересекаются или касаются — возвращает 0.0.
        Если перекрываются по одной оси — возвращает чистый зазор по другой.
        Если находятся по диагонали — евклидово расстояние между ближайшими углами.

        Args:
            bbox1 (list[float]): Первый bounding box [x1, y1, x2, y2].
            bbox2 (list[float]): Второй bounding box [x1, y1, x2, y2].

        Returns:
            float: Расстояние между границами bbox в пикселях.
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Вычисление зазора по горизонтали (dx)
        if x1_max < x2_min:
            dx = x2_min - x1_max  # bbox2 справа от bbox1
        elif x2_max < x1_min:
            dx = x1_min - x2_max  # bbox2 слева от bbox1
        else:
            dx = 0  # Перекрываются по X

        # Вычисление зазора по вертикали (dy)
        if y1_max < y2_min:
            dy = y2_min - y1_max  # bbox2 снизу от bbox1
        elif y2_max < y1_min:
            dy = y1_min - y2_max  # bbox2 сверху от bbox1
        else:
            dy = 0  # Перекрываются по Y

        # Если перекрываются и по X, и по Y — расстояние 0
        if dx == 0 and dy == 0:
            return 0.0

        # Если перекрываются по одной из осей — возвращаем чистый зазор по другой
        if dx == 0:
            return float(dy)
        if dy == 0:
            return float(dx)

        # Если находятся по диагонали — евклидово расстояние между ближайшими углами
        return math.sqrt(dx * dx + dy * dy)

    def calculate_distance(self, node1_id, node2_id):
        """
        Вычисляет расстояние между двумя узлами графа.

        В текущей реализации используется расстояние между границами bbox
        (calculate_bbox_distance). Закомментированный вариант — расстояние
        между центрами (math.hypot).

        Args:
            node1_id (int): Идентификатор первого узла.
            node2_id (int): Идентификатор второго узла.

        Returns:
            float: Расстояние между узлами в пикселях.
        """
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]

        # Старый вариант (расстояние между центрами):
        # dx = node1['center'][0] - node2['center'][0]
        # dy = node1['center'][1] - node2['center'][1]
        # return math.hypot(dx, dy)

        # Новый вариант (расстояние между границами bbox):
        return self.calculate_bbox_distance(node1['bbox'], node2['bbox'])

    def add_edge(self, node1_id, node2_id, connection_type):
        """
        Добавляет ребро между двумя узлами и регистрирует связь в обоих узлах.

        Args:
            node1_id (int): Идентификатор первого узла.
            node2_id (int): Идентификатор второго узла.
            connection_type (str): Тип связи (например, 'exclusive_adaptive').

        Returns:
            bool: True при успешном добавлении.
        """
        distance = self.calculate_distance(node1_id, node2_id)
        self.edges.append((node1_id, node2_id, connection_type, distance))
        self.nodes[node1_id]['connections'].append((node2_id, connection_type, distance))
        self.nodes[node2_id]['connections'].append((node1_id, connection_type, distance))
        return True

    def get_vertical_overlap(self, elem_bbox, text_bbox):
        """
        Вычисляет долю вертикального перекрытия текста относительно элемента.

        Args:
            elem_bbox (list[float]): Bounding box элемента.
            text_bbox (list[float]): Bounding box текста.

        Returns:
            float: Доля перекрытия по вертикали (0..1) относительно высоты элемента.
        """
        y1_1, y2_1 = elem_bbox[1], elem_bbox[3]
        y1_2, y2_2 = text_bbox[1], text_bbox[3]

        overlap_y = max(0.0, min(y2_1, y2_2) - max(y1_1, y1_2))
        height1 = y2_1 - y1_1
        return overlap_y / height1 if height1 > 0 else 0.0

    def get_horizontal_overlap(self, elem_bbox, text_bbox):
        """
        Вычисляет долю горизонтального перекрытия между элементом и текстом.

        Сначала нормирует на ширину элемента; если ширина элемента нулевая,
        нормирует на ширину текста.

        Args:
            elem_bbox (list[float]): Bounding box элемента.
            text_bbox (list[float]): Bounding box текста.

        Returns:
            float: Доля перекрытия по горизонтали (0..1).
        """
        x1_1, x2_1 = elem_bbox[0], elem_bbox[2]
        x1_2, x2_2 = text_bbox[0], text_bbox[2]

        overlap_x = max(0.0, min(x2_1, x2_2) - max(x1_1, x1_2))
        width1 = x2_1 - x1_1

        if width1 > 0:
            return overlap_x / width1

        width2 = x2_2 - x1_2
        return (overlap_x / width2) if width2 > 0 else 0.0

    def get_horizontal_direction(self, elem_bbox, text_bbox):
        """
        Определяет горизонтальное направление текста относительно элемента.

        Args:
            elem_bbox (list[float]): Bounding box элемента.
            text_bbox (list[float]): Bounding box текста.

        Returns:
            str: 'right', если центр текста правее центра элемента, иначе 'left'.
        """
        elem_cx = (elem_bbox[0] + elem_bbox[2]) / 2.0
        text_cx = (text_bbox[0] + text_bbox[2]) / 2.0
        return 'right' if text_cx > elem_cx else 'left'

    def get_direction_zone(self, elem_bbox, text_bbox):
        """
        Определяет зону направления текста относительно элемента.

        Возвращает словарь с горизонтальной и вертикальной зонами, основной
        зоной (по углу), углом в градусах [0, 360) и смещениями dx, dy между
        центрами.

        Основная зона определяется по углу:
            - right: угол <= 45 или >= 315
            - bottom: 45 < угол < 135
            - left: 135 <= угол <= 225
            - top: иначе

        Args:
            elem_bbox (list[float]): Bounding box элемента.
            text_bbox (list[float]): Bounding box текста.

        Returns:
            dict: {
                'horizontal': str ('left'|'right'),
                'vertical': str ('top'|'bottom'),
                'primary': str ('left'|'right'|'top'|'bottom'),
                'angle': float,
                'dx': float,
                'dy': float
            }
        """
        elem_cx = (elem_bbox[0] + elem_bbox[2]) / 2.0
        elem_cy = (elem_bbox[1] + elem_bbox[3]) / 2.0
        text_cx = (text_bbox[0] + text_bbox[2]) / 2.0
        text_cy = (text_bbox[1] + text_bbox[3]) / 2.0

        dx = text_cx - elem_cx
        dy = text_cy - elem_cy

        horiz_zone = 'right' if dx > 0 else 'left'
        vert_zone = 'bottom' if dy > 0 else 'top'

        # Calculate standard mathematical angle in degrees [0, 360)
        angle = math.degrees(math.atan2(dy, dx)) % 360

        if angle <= 45 or angle >= 315:
            primary_zone = 'right'
        elif 45 < angle < 135:
            primary_zone = 'bottom'
        elif 135 <= angle <= 225:
            primary_zone = 'left'
        else:
            primary_zone = 'top'

        return {
            'horizontal': horiz_zone,
            'vertical': vert_zone,
            'primary': primary_zone,
            'angle': angle,
            'dx': dx,
            'dy': dy
        }

    def get_direction_string(self, direction_info):
        """
        Преобразует угол направления в текстовую метку из 8 направлений.

        Args:
            direction_info (dict): Словарь, полученный из get_direction_zone,
                должен содержать ключ 'angle'.

        Returns:
            str: Одно из 'right', 'bottom-right', 'bottom', 'bottom-left',
                 'left', 'top-left', 'top', 'top-right'.
        """
        angle = direction_info['angle']

        if 337.5 <= angle or angle < 22.5:
            return "right"
        if 22.5 <= angle < 67.5:
            return "bottom-right"
        if 67.5 <= angle < 112.5:
            return "bottom"
        if 112.5 <= angle < 157.5:
            return "bottom-left"
        if 157.5 <= angle < 202.5:
            return "left"
        if 202.5 <= angle < 247.5:
            return "top-left"
        if 247.5 <= angle < 292.5:
            return "top"
        return "top-right"

    def calculate_connection_score_simple(self, element_id, text_id, max_distance, direction_info, vertical_overlap):
        """
        Вычисляет интегральный скор для пары «элемент — текст».

        Компоненты скора:
            - distance_score: 1 - distance / max_distance (не меньше 0);
            - overlap_score: min(1.0, vertical_overlap * 3.0);
            - direction_score: множитель в зависимости от основной зоны
              (right ×1.15, left ×1.10, top/bottom ×1.05);
            - бонус ×1.2, если расстояние < 20% от max_distance.

        Итоговый скор ограничен сверху 1.0.

        Args:
            element_id (int): Идентификатор элемента.
            text_id (int): Идентификатор текстового узла.
            max_distance (float): Максимально допустимая дистанция.
            direction_info (dict): Результат get_direction_zone.
            vertical_overlap (float): Доля вертикального перекрытия (0..1).

        Returns:
            float: Скор в диапазоне [0.0, 1.0].
        """
        distance = self.calculate_distance(element_id, text_id)

        distance_score = max(0.0, 1.0 - (distance / max_distance))
        overlap_score = min(1.0, vertical_overlap * 3.0)

        direction_score = 1.0
        if direction_info['primary'] == 'right':
            direction_score *= 1.15
        elif direction_info['primary'] == 'left':
            direction_score *= 1.10
        elif direction_info['primary'] in ['top', 'bottom']:
            direction_score *= 1.05

        total_score = (distance_score * 0.6 + overlap_score * 0.3 + 0.1) * direction_score

        if distance < max_distance * 0.2:
            total_score *= 1.2

        return min(1.0, total_score)