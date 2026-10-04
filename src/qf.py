from pydantic import BaseModel, field_validator


class QF(BaseModel):
    """
    Pydantic-модель автоматического выключателя (QF).

    Описывает один автоматический выключатель и его ключевые атрибуты,
    извлеченные из схемы. Все строковые поля по умолчанию пустые,
    чтобы модель можно было создавать даже при частично распознанных данных.

    Attributes:
        ID_QF (str): Уникальный идентификатор автомата (например, "QF1").
        Name (str): Наименование/маркировка автомата.
        Current (str): Номинальный ток.
        Voltage (str): Номинальное напряжение.
        Current_Close (str): Ток уставки/отключения (или иной параметр "close").
        Polus (str): Количество полюсов.
        Mounting_Type (str): Тип монтажа ("0" или "1"); по умолчанию "0".
    """

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
        """
        Валидатор строковых полей: заменяет None на пустую строку.

        Pydantic по умолчанию не пропускает None для полей типа str,
        поэтому валидатор делает модель устойчивой к отсутствующим значениям
        (None -> "").

        Args:
            v (str | None): Значение поля, поступившее в модель.

        Returns:
            str: Исходное значение, если оно не None, иначе пустая строка.
        """
        return v if v is not None else ""

    def print_data(self) -> None:
        """
        Выводит данные автомата в консоль одной строкой.

        Формат вывода:
            "<ID_QF>: <Name> <Current> <Voltage> <Current_Close> <Polus> <Mounting_Type>"

        Returns:
            None
        """
        print(f"{self.ID_QF}: {self.Name} {self.Current} {self.Voltage} {self.Current_Close} {self.Polus} {self.Mounting_Type}")


def create_qf(id_qf: str, name: str, current: str, voltage: str, current_close: str, polus: str, mounting_type: str) -> QF:
    """
    Создает объект QF через конструктор Pydantic.

    Является удобной фабричной оберткой над QF(...), принимающей все
    атрибуты позиционно и возвращающей готовую модель.

    Args:
        id_qf (str): Уникальный идентификатор автомата.
        name (str): Наименование/маркировка.
        current (str): Номинальный ток.
        voltage (str): Номинальное напряжение.
        current_close (str): Ток уставки/отключения.
        polus (str): Количество полюсов.
        mounting_type (str): Тип монтажа ("0" или "1").

    Returns:
        QF: Созданный объект автоматического выключателя.
    """
    return QF(
        ID_QF=id_qf,
        Name=name,
        Current=current,
        Voltage=voltage,
        Current_Close=current_close,
        Polus=polus,
        Mounting_Type=mounting_type
    )