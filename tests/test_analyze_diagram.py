import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
from search_object import analyze_circuit_diagram, YOLOClassID

def test_analyze_circuit_diagram_orchestration(mocker):
    """Интеграционный тест для проверки пайплайна analyze_circuit_diagram с моками"""
    
    # 1. Фейковые входные данные (результаты от YOLO)
    mock_results = [{
        'results': [
            # Текст
            [10, 10, 50, 30, 0.9, float(YOLOClassID.TEXT), 30, 20],
            # Автомат
            [10, 50, 50, 100, 0.9, float(YOLOClassID.AUTOMATIC), 30, 75]
        ],
        'names': {0: 'automatic', 7: 'text'}
    }]
    
    # Пустая картинка-заглушка (черный квадрат 200x200)
    dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)

    # 2. МОК (подделка) тяжелых внешних функций
    # Подделка ответ text-сервиса
    mock_recognize = mocker.patch('search_object.recognize_with_text_service')
    mock_recognize.return_value = (["Mocked_QF_Object"], []) # возвращает фейковые списки поиска и количества
    
    # Подделка визуализации
    mocker.patch('search_object.render_circuit_visualization')

    # 3. Запуск тестируемой функции
    list_for_searching, list_for_quantity, mounting_types_dict = analyze_circuit_diagram(mock_results, dummy_image)

    # 4. Проверка результата работы пайплайна
    # Ожидание, что функция успешно прошла все шаги и вернула данные из нашего мока
    assert len(list_for_searching) == 1
    assert list_for_searching[0] == "Mocked_QF_Object"
    
    #  Мок сети действительно вызывался?
    mock_recognize.assert_called_once()

