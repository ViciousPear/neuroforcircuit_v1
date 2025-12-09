from pydantic import BaseModel, field_validator


class QF(BaseModel):
    ID_QF: str = ""
    Name: str = ""
    Current: str = ""
    Voltage: str = ""
    Current_Close: str = ""
    Polus: str = ""
    Mounting_Type: str = "0"

    

    @field_validator("Name", "Current", "Voltage", "Current_Close", "Polus", "Mounting_Type")
    @classmethod
    def validate_empty(cls, v: str) -> str:
        """Заменяет None на пустую строку."""
        return v if v is not None else ""

    def print_data(self) -> None:
        """Выводит данные в консоль."""
        print(f"{self.ID_QF}: {self.Name} {self.Current} {self.Voltage} {self.Current_Close} {self.Polus} {self.Mounting_Type}")

def create_qf(id_qf: str, name: str, current: str, voltage: str, current_close: str, polus: str, mounting_type: str) -> QF:
    """Создаёт объект QF. Теперь просто вызывает конструктор Pydantic."""
    return QF(
        ID_QF=id_qf,
        Name=name,
        Current=current,
        Voltage=voltage,
        Current_Close=current_close,
        Polus=polus,
        Mounting_Type=mounting_type
    )
