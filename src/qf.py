from pydantic import BaseModel, field_validator


class QF(BaseModel):
    ID_QF: str = ""
    Current: str = ""
    Voltage: str = ""
    Current_Close: str = ""

    

    @field_validator("Current", "Voltage", "Current_Close")
    @classmethod
    def validate_empty(cls, v: str) -> str:
        """Заменяет None на пустую строку."""
        return v if v is not None else ""

    def print_data(self) -> None:
        """Выводит данные в консоль."""
        print(f"{self.ID_QF}: {self.Current} {self.Voltage} {self.Current_Close}")

def create_qf(id_qf: str, current: str, voltage: str, current_close: str) -> QF:
    """Создаёт объект QF. Теперь просто вызывает конструктор Pydantic."""
    return QF(
        ID_QF=id_qf,
        Current=current,
        Voltage=voltage,
        Current_Close=current_close
    )
