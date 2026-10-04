import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from search_object import results_boxes, YOLOClassID

def test_results_boxes_filtering_and_sorting():
    """Тест проверяет, что results_boxes отсекает слабые предсказания 
    и правильно распределяет боксы по категориям классов"""
    
    # Имитируем ответ от YOLO-модели в виде словаря
    mock_results = [{
        'results': [
            # 1. Текст (cls_id = 7), conf = 0.9 (пройдет порог)
            [10.0, 20.0, 50.0, 60.0, 0.9, float(YOLOClassID.TEXT)],
            
            # 2. Автомат (cls_id = 0), conf = 0.3 (НЕ пройдет порог 0.5, должен отсеяться)
            [100.0, 100.0, 150.0, 150.0, 0.3, float(YOLOClassID.AUTOMATIC)],
            
            # 3. Автомат (cls_id = 0), conf = 0.85 (пройдет порог)
            [200.0, 200.0, 250.0, 300.0, 0.85, float(YOLOClassID.AUTOMATIC)]
        ],
        'names': {0: 'automatic', 7: 'text'}
    }]
    
    score_threshold = 0.5
    
    # Запуск функцию
    text_boxes, automatics_boxes, other_boxes, transformer_boxes, rcd_boxes, bracing_boxes = results_boxes(mock_results, score_threshold)
    
    # 1. Должен остаться ровно 1 текстовый бокс
    assert len(text_boxes) == 1
    # Проверrf координат и вычисленного центра для текста: cx = (10+50)/2 = 30.0, cy = (20+60)/2 = 40.0
    assert text_boxes[0][:6] == (10, 20, 50, 60, 0.9, YOLOClassID.TEXT)
    assert text_boxes[0][6] == 30.0
    assert text_boxes[0][7] == 40.0
    
    # 2. Слабый автомат отсеялся, сильный остался -> всего 1 автомат
    assert len(automatics_boxes) == 1
    assert automatics_boxes[0][4] == 0.85 # conf
    
    # 3. Остальные списки должны быть пустыми
    assert len(transformer_boxes) == 0
    assert len(rcd_boxes) == 0
    assert len(bracing_boxes) == 0
    assert len(other_boxes) == 0