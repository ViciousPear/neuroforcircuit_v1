import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
from search_object import recognize_only, YOLOClassID



'search_object.cv2.imread'
'search_object.send_to_yolo_service'
'search_object.analyze_circuit_diagram'
def test_recognize_only_pipeline(mock_analyze, mock_send_yolo, mock_imread):
    """Интеграционный тест для проверки главного пайплайна recognize_only"""
    
    # 1. Подделываем чтение картинки через OpenCV (возвращаем пустую матрицу-заглушку)
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img
    
    # 2. Подделываем ответ от YOLO-сервиса (например, нашелся один автомат класса 0)
    mock_send_yolo.return_value = {
        'results': [[10, 10, 50, 50, 0.9, 0, 30, 30]],
        'names': {0: 'automatic'}
    }
    
    # 3. Подделываем результат работы analyze_circuit_diagram 
    # (возвращает пустые списки для поиска и словарь типов монтажа)
    mock_analyze.return_value = ([], [], {1: "0"})
    
    # 4. Запускаем тестируемую функцию
    image, detected_elements = recognize_only("fake_path/schema.png")
    
    # 5. Проверяем результаты:
    # Убеждаемся, что функция вернула картинку и список элементов
    assert image is not None
    assert isinstance(detected_elements, list)
    
    # Проверяем, что в списке обнаружился автомат (QF)
    qf_elements = [e for e in detected_elements if e['type'] == 'QF']
    assert len(qf_elements) == 1
    assert qf_elements[0]['number'] == 1
    
    # Проверяем, что все ключевые шаги пайплайна действительно вызывались
    mock_imread.assert_called_once_with("fake_path/schema.png")
    mock_send_yolo.assert_called_once()
    mock_analyze.assert_called_once()