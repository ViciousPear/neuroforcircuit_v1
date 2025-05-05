import re
from pydantic import BaseModel, field_validator

class Trans_TA(BaseModel):
    ta_name: str = ""
    ta_quantity: int = 0


    @field_validator("ta_quantity")
    def validate_quantity(cls, v):
        if v < 0:
           v = 1
        return v

    def __init__(self, ta_text: str):
        # Инициализация Pydantic
        super().__init__()  
        self.ta_name = ta_text
        # Парсинг при создании
        self.ta_quantity = self._parse_quantity(ta_text)  

    def _parse_quantity(self, ta_text: str) -> int:
        """Приватный метод для парсинга количества из текста."""
        pattern = r"\d*[ТT][АA]+[A-Za-z]?(\d+)"
        matches = re.findall(pattern, ta_text)
        
        if not matches:
            return 0  
        
        numbers = [int(num) for num in matches]
        return max(numbers) - min(numbers)


def create_ta(ta_text: str) -> Trans_TA:
    """Создаёт объект Trans_TA с автоматическим парсингом."""
    return Trans_TA(ta_text)



