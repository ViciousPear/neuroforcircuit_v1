import fastapi
from src import qf, ta, search_object
from database import connection, searching_in_base
from typing import List
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)
class DatabaseAPI:
    """API для взаимодействия бизнес-логики с базой данных"""
    
    @staticmethod
    def search_qf_breakers(qf_object: qf.QF) -> List[dict]:
        """
        Поиск автоматических выключателей в БД
        Args:
            qf_object: Объект с параметрами поиска
        Returns:
            Список найденных записей
        """
        
        try:
            search_params = {
                "ID_QF": qf_object.ID_QF,
                "Current": qf_object.Current,
                "Voltage": qf_object.Voltage,
                "Current_Close": qf_object.Current_Close
            }
            result = searching_in_base.search_automatics(search_params)
            return result
        except Exception as e:
            logger.error(f"Ошибка при поиске выключателей: {e}")
            return []
    
    @staticmethod
    def search_counters(limit: int = 10) -> List[dict]:
        """
        Поиск счетчиков в БД
        Args:
            limit: Ограничение количества результатов
        Returns:
            Список найденных записей
        """
        
        try:
            result = searching_in_base.search_wh()
            return result
        except Exception as e:
            logger.error(f"Ошибка при поиске трансформаторов: {e}")
            return []
        
    
    @staticmethod
    def search_transformators(limit: int = 10) -> List[dict]:
        """
        Поиск трансформаторов в БД
        Args:
            limit: Ограничение количества результатов
        Returns:
            Список найденных записей
        """
        
        try:
            result = searching_in_base.search_trans()
            return result
        except Exception as e:
            logger.error(f"Ошибка при поиске трансформаторов: {e}")
            return []
        

