import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from graph_mapper import create_graphs, create_a_mind_map

def test_create_graphs_populates_nodes():
    """Тест проверяет, что create_graphs корректно добавляет узлы разного типа в граф"""
    
    # 1. Мок графа
    mock_graph = MagicMock()
    # Метод add_node будет возвращать фейковый ID для каждого добавленного бокса
    mock_graph.add_node.side_effect = ['node_auto_1', 'node_trans_1', 'node_text_1']
    mock_graph.nodes = {}

    # 2. Nестовые данные (боксики)
    # Формат: (x1, y1, x2, y2, conf, cls_id, cx, cy)
    automotive_boxes = [(10, 10, 30, 30, 0.9, 0, 20, 20)]
    transformer_boxes = [(40, 10, 60, 30, 0.9, 1, 50, 20)]
    rcd_boxes = []
    text_boxes = [(70, 10, 90, 30, 0.9, 7, 80, 20)]

    # 3. Тестируемая функция
    node_ids, text_nodes = create_graphs(
        mock_graph, automotive_boxes, transformer_boxes, rcd_boxes, text_boxes
    )

    # 4. Проверка результатов:
    # Узлы должны быть добавлены в словари mappings
    assert len(node_ids) == 2  # автомат и трансформатор
    assert len(text_nodes) == 1  # текст
    
    # Проверка, что метод add_node вызывался нужное количество раз (по числу боксов)
    assert mock_graph.add_node.call_count == 3
    
    # Проверка, что адаптивные связи для типов тоже вызывались
    mock_graph.create_exclusive_connections_adaptive.assert_any_call('automatic')
    mock_graph.create_exclusive_connections_adaptive.assert_any_call('transformer')

def test_create_a_mind_map_structure():
    """Тест проверяет структуру словарей, возвращаемых функцией create_a_mind_map"""
    
    mock_graph = MagicMock()
    # Имитация отсутствия ребер, чтобы проверить работу пустой карты или базовой структуры
    mock_graph.edges = []
    
    text_boxes = [(10, 10, 20, 20, 0.9, 7, 15, 15)]
    automatics_boxes = [(30, 10, 50, 30, 0.9, 0, 40, 20)]
    node_ids = {tuple(automatics_boxes[0]): 'node_auto_1'}
    text_nodes = {'node_text_1': text_boxes[0]}
    
    t2a, a2t, maps = create_a_mind_map(
        mock_graph, text_boxes, automatics_boxes, node_ids, text_nodes
    )
    
    # Проверка, что функция возвращает словари нужного формата
    assert isinstance(t2a, dict)
    assert isinstance(a2t, dict)
    assert isinstance(maps, dict)
    assert 'automatic' in maps
    assert 'transformer' in maps
    assert 'rcd' in maps