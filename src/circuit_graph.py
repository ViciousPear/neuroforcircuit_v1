import math

class CircuitGraph:
    def __init__(self, distance_multiplier=1.7):
        self.nodes = {}
        self.edges = []
        self.next_id = 0
        self.use_orientation = False
        self.distance_multiplier = distance_multiplier  # Коэффициент для расстояния
    
    def calculate_element_size(self, bbox):
        """Вычисляет реальный размер элемента на основе bbox"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height
    
    def calculate_adaptive_max_distance(self, bbox):
        """
        Рассчитывает максимальное расстояние как размер элемента * коэффициент.
        """
        width, height = self.calculate_element_size(bbox)
        
        # Используем диагональ элемента как основной размер
        diagonal = math.sqrt(width**2 + height**2)
        
        # Максимальное расстояние = диагональ * коэффициент
        adaptive_distance = diagonal * self.distance_multiplier
        
        # Минимальное и максимальное ограничения
        min_distance = 50  # Минимальное расстояние в пикселях
        max_distance = 200  # Максимальное расстояние в пикселях
        
        adaptive_distance = max(min_distance, min(adaptive_distance, max_distance))
        
        # Логирование для отладки
        print(f"🔧 Адаптивное расстояние: размер={width:.0f}x{height:.0f}, "
              f"диагональ={diagonal:.0f}px, "
              f"коэффициент={self.distance_multiplier}, "
              f"расстояние={adaptive_distance:.1f}px")
        
        return adaptive_distance
    
    def create_exclusive_connections_adaptive(self, element_type='automatic'):
        """
        Адаптивный метод связывания: расстояние зависит от размера элемента.
        """
        print(f"🔗 Создаю АДАПТИВНЫЕ связи для {element_type} "
              f"(коэффициент расстояния: {self.distance_multiplier})...")
        
        element_nodes = []
        text_nodes = []
        
        for node_id, node_data in self.nodes.items():
            if node_data['type'] == element_type:
                element_nodes.append(node_id)
            elif node_data['type'] == 'text':
                text_nodes.append(node_id)
        
        # Создаем связи для каждого элемента с его адаптивным расстоянием
        used_elements = set()
        used_texts = set()
        connections_created = 0
        
        # Обрабатываем каждый элемент отдельно
        for element_id in element_nodes:
            element_data = self.nodes[element_id]
            
            # Рассчитываем адаптивное расстояние для этого элемента
            adaptive_max_distance = self.calculate_adaptive_max_distance(
                element_data['bbox']
            )
            
            # Собираем возможные тексты для этого элемента
            possible_pairs = []
            
            for text_id in text_nodes:
                if text_id in used_texts:
                    continue
                    
                text_data = self.nodes[text_id]
                distance = self.calculate_distance(element_id, text_id)
                
                if distance <= adaptive_max_distance:
                    vertical_overlap = self.get_vertical_overlap(
                        element_data['bbox'], text_data['bbox']
                    )
                    
                    # Более мягкое требование к вертикальному перекрытию для трансформаторов
                    overlap_threshold = 0.1 if element_type == 'transformer' else 0.2
                    
                    if vertical_overlap > overlap_threshold:
                        score = self.calculate_connection_score(
                            element_id, text_id, adaptive_max_distance
                        )
                        
                        possible_pairs.append({
                            'text_id': text_id,
                            'distance': distance,
                            'score': score,
                            'text_content': text_data.get('text', 'Unknown'),
                            'vertical_overlap': vertical_overlap
                        })
            
            # Сортируем по score и выбираем лучший
            if possible_pairs:
                possible_pairs.sort(key=lambda x: x['score'], reverse=True)
                best_pair = possible_pairs[0]
                
                # Создаем связь
                self.add_edge(element_id, best_pair['text_id'], 'exclusive_adaptive')
                used_elements.add(element_id)
                used_texts.add(best_pair['text_id'])
                connections_created += 1
                
                direction = self.get_horizontal_direction(
                    element_data['bbox'], 
                    self.nodes[best_pair['text_id']]['bbox']
                )
                
                width, height = self.calculate_element_size(element_data['bbox'])
                print(f"  ✅ {element_type} (размер: {width:.0f}x{height:.0f}px) → "
                      f"'{best_pair['text_content']}' "
                      f"(дистанция: {best_pair['distance']:.1f}px из {adaptive_max_distance:.1f}px, "
                      f"позиция: {direction})")
            else:
                # Логируем элементы без текста
                width, height = self.calculate_element_size(element_data['bbox'])
                print(f"  ⚠️ {element_type} (размер: {width:.0f}x{height:.0f}px) - не найден текст "
                      f"(макс. дистанция: {adaptive_max_distance:.1f}px)")
        
        # Статистика
        print(f"✅ Создано {connections_created} адаптивных связей")
        
        if len(element_nodes) > connections_created:
            print(f"  ⚠️ {len(element_nodes) - connections_created} элементов остались без текста")
        
        return connections_created

    # ... остальные методы остаются без изменений ...
    
    def add_node(self, bbox, node_type, confidence, text_content=None):
        node_id = self.next_id
        self.next_id += 1
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
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
    
    def calculate_distance(self, node1_id, node2_id):
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        dx = node1['center'][0] - node2['center'][0]
        dy = node1['center'][1] - node2['center'][1]
        return math.sqrt(dx*dx + dy*dy)
    
    def add_edge(self, node1_id, node2_id, connection_type):
        distance = self.calculate_distance(node1_id, node2_id)
        self.edges.append((node1_id, node2_id, connection_type, distance))
        self.nodes[node1_id]['connections'].append((node2_id, connection_type, distance))
        self.nodes[node2_id]['connections'].append((node1_id, connection_type, distance))
        return True
    
    def get_vertical_overlap(self, elem_bbox, text_bbox):
        """Вычисляет вертикальное перекрытие между двумя bbox"""
        y1_1, y2_1 = elem_bbox[1], elem_bbox[3]
        y1_2, y2_2 = text_bbox[1], text_bbox[3]
        
        overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        height1 = y2_1 - y1_1
        return overlap_y / height1 if height1 > 0 else 0
    
    def get_horizontal_direction(self, elem_bbox, text_bbox):
        """Определяет горизонтальное направление текста относительно элемента"""
        elem_cx = (elem_bbox[0] + elem_bbox[2]) / 2
        text_cx = (text_bbox[0] + text_bbox[2]) / 2
        
        if text_cx > elem_cx:
            return 'right'
        else:
            return 'left'
    
    def calculate_connection_score(self, element_id, text_id, max_distance=50):
        """Упрощенная оценка связи - только расстояние и вертикальное выравнивание"""
        element = self.nodes[element_id]
        text = self.nodes[text_id]
        
        distance = self.calculate_distance(element_id, text_id)
        vertical_overlap = self.get_vertical_overlap(element['bbox'], text['bbox'])
        
        # Баллы за расстояние (чем ближе - тем лучше)
        distance_score = max(0, 1 - (distance / max_distance))
        
        # Баллы за вертикальное перекрытие
        overlap_score = min(1.0, vertical_overlap * 2)
        
        # Баллы за горизонтальное положение (небольшой бонус)
        direction = self.get_horizontal_direction(element['bbox'], text['bbox'])
        direction_bonus = 1.1 if direction == 'right' else 1.0  # Небольшой бонус для правой стороны
        
        # Общий score
        total_score = (distance_score * 0.6 + overlap_score * 0.4) * direction_bonus
        
        return min(1.0, total_score)