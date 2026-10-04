import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from search_object import numeric_automatics_with_mounting, numeric_other_elements

def test_numeric_automatics_with_mounting():
    """Тест проверяет правильность нумерации автоматов с учетом привязанных текстов и типа монтажа"""
    
    # Имитация отсортированные текстовые блоки
    text_box_1 = (10, 10, 30, 30, 0.9, 7, 20, 20)
    text_box_2 = (10, 100, 30, 120, 0.9, 7, 20, 110)
    text_boxes_sorted = [text_box_1, text_box_2]
    
    # Автоматы
    auto_box_1 = (50, 10, 90, 40, 0.9, 0, 70, 25)
    auto_box_2 = (50, 100, 90, 130, 0.9, 0, 70, 115)
    auto_box_3 = (50, 200, 90, 230, 0.9, 0, 70, 215) # Этот автомат будет без текста
    automatics_boxes = [auto_box_1, auto_box_2, auto_box_3]
    
    # Карта связей: text_box_1 ссылается на auto_box_2, text_box_2 - на auto_box_1
    text_to_automatic_map = {
        tuple(text_box_1): auto_box_2,
        tuple(text_box_2): auto_box_1
    }
    
    text_numbers = {
        text_box_1: 1,
        text_box_2: 2
    }
    
    automatic_mounting_data = {
        auto_box_1: ("0", auto_box_1[:4]),
        auto_box_2: ("1", auto_box_2[:4]),
        auto_box_3: ("0", auto_box_3[:4])
    }
    
    # Запуск функции
    box_numbers, automatics_with_texts, current_number = numeric_automatics_with_mounting(
        text_boxes_sorted, text_to_automatic_map, automatics_boxes, text_numbers, automatic_mounting_data
    )
    
    # 1. Первым обрабатывается text_box_1, значит связанный с ним auto_box_2 должен получить номер 1
    assert box_numbers[auto_box_2] == 1
    
    # 2. Вторым идет text_box_2, связанный с auto_box_1, он получает номер 2
    assert box_numbers[auto_box_1] == 2
    
    # 3. Автомат без текста (auto_box_3) получает номер 3
    assert box_numbers[auto_box_3] == 3
    
    # 4. Следующий свободный номер должен быть 4
    assert current_number == 4

    def test_numeric_other_elements():
        """Тест проверяет совместную сортировку и нумерацию трансформаторов, УЗО и прочих элементов[span_4](start_span)[span_4](end_span)"""
        
        # Формат бокса: (x1, y1, x2, y2, conf, cls_id, cx, cy)
        
        # 1. УЗО выше по Y (y=50, x=20)
        rcd_box = (10, 40, 30, 60, 0.9, 9, 20, 50)
        
        # 2. Другой элемент выше по Y, но правее (y=50, x=100)
        other_box = (90, 40, 110, 60, 0.9, 3, 100, 50)
        
        # 3. Трансформатор ниже по Y (y=150, x=20)
        trans_box = (10, 140, 30, 160, 0.9, 1, 20, 150)
        
        transformer_boxes = [trans_box]
        rcd_boxes = [rcd_box]
        other_boxes = [other_box]
        
        box_numbers = {}
        current_number = 3  # Допустим, автоматы заняли первые два номера (1 и 2)
        
        # Запускаем функцию
        sorted_elements = numeric_other_elements(
            transformer_boxes, rcd_boxes, other_boxes, box_numbers, current_number
        )
        
        # 1. Порядок сортировки (сначала по Y, потом по X):
        # Сначала rcd_box (y=50, x=20), затем other_box (y=50, x=100), затем trans_box (y=150, x=20)
        assert sorted_elements[0] == rcd_box
        assert sorted_elements[1] == other_box
        assert sorted_elements[2] == trans_box
        
        # 2. Проверка присвоения сквозных номеров начиная с current_number (3)
        assert box_numbers[rcd_box] == 3
        assert box_numbers[other_box] == 4
        assert box_numbers[trans_box] == 5