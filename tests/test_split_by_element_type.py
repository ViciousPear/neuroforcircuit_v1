import sys
import os
from collections import Counter

# Добавляем путь к папке source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from search_object import split_by_element_type

def test_split_by_element_type():
    """Тест проверяет корректность извлечения количества трансформаторов и счетчиков из Counter"""
    
    # Имитация данных счетчика классов YOLO:
    # Класс 0 (автоматы) = 3 штуки
    # Класс 1 (трансформаторы) = 2 штуки
    # Класс 2 (счетчики) = 1 штука
    mock_class_counts = Counter({
        0.0: 3,
        1.0: 2,
        2.0: 1
    })
    
    list_trans, list_wh = split_by_element_type(mock_class_counts)
    
    # Проверка, что функция вернула ожидаемые списки с количеством
    assert list_trans == [2]
    assert list_wh == [1]

def test_split_by_element_type_empty():
    """Тест проверяет работу функции, если нужные классы вообще отсутствуют"""
    mock_class_counts = Counter({0.0: 5})
    
    list_trans, list_wh = split_by_element_type(mock_class_counts)
    
    assert list_trans == [0]
    assert list_wh == [0]