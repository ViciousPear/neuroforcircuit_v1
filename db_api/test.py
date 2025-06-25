import pytest
from database_api import DatabaseAPI
from src.qf import QF

def test_search_circuit_breakers():
    qf_obj = QF(ID_QF="", Current="16A", Voltage="230V", Current_Close="")
    result = DatabaseAPI.search_qf_breakers(qf_obj)
    assert isinstance(result, list)