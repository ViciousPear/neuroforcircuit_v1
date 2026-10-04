import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from geometry import calculate_horizontal_overlap, determine_mounting_type_and_bbox

def test_calculate_horizontal_overlap_full():
    box1 = (10, 10, 50, 50)
    box2 = (10, 20, 50, 60)

    overlap = calculate_horizontal_overlap(box1, box2)

    assert overlap == 1.0

def test_calculate_horizontal_overlap_none():
    box1 = (10, 10, 20, 50)
    box2 = (30, 10, 40, 50)

    overlap = calculate_horizontal_overlap(box1, box2)

    assert overlap == 0.0

def test_determine_mounting_type_zero():
    """Тест: монтаж должен быть равен '0', если рядом нет крепежа (bracing)"""
    # Автомат по центру: x1=10, y1=50, x2=30, y2=100
    # Формат бокса для функции: (x1, y1, x2, y2, conf, cls_id, cx, cy)
    automatic_box = (10, 50, 30, 100, 0.9, 0, 20, 75)
    
    # Пустой список крепежа
    bracing_boxes = []
    
    mounting_type, final_bbox = determine_mounting_type_and_bbox(automatic_box, bracing_boxes)
    
    assert mounting_type == "0"
    # При типе "0" возвращаются первые 4 координаты исходного автомата
    assert final_bbox == (10, 50, 30, 100)

def test_determine_mounting_type_one():
    """Тест: монтаж равен '1', если есть крепеж и сверху, и снизу"""
    automatic_box = (10, 50, 30, 100, 0.9, 0, 20, 75)
    
    # Крепеж сверху (y до 50) и снизу (y от 100) с хорошим горизонтальным перекрытием
    bracing_boxes = [
        (10, 30, 30, 45, 0.9, 4, 20, 37.5),   
        (10, 105, 30, 120, 0.9, 4, 20, 112.5)  
    ]
    
    mounting_type, final_bbox = determine_mounting_type_and_bbox(automatic_box, bracing_boxes)
    
    assert mounting_type == "1"
    # При типе "1" итоговый bbox должен охватывать крайние точки всех элементов (от минимума по Y до максимума по Y)
    # Минимальный Y здесь 30 (от верхнего крепежа), максимальный Y — 120 (от нижнего)
    assert final_bbox == (10, 30, 30, 120)

