import re


class Trans_TA:
    """
    Модель трансформатора тока (TA), вычисляющая количество трансформаторов
    по текстовой маркировке.

    Класс инкапсулирует имя (маркировку) трансформатора тока и вычисляет
    количество трансформаторов на основе чисел, извлеченных из текста
    (например, "TA1, TA3" → 2).

    Приватные атрибуты:
        __ta_name (str): Имя/маркировка трансформатора тока.
        __ta_quantity (int): Вычисленное количество трансформаторов.

    Публичный интерфейс:
        - ta_name (property + setter): доступ к имени.
        - ta_quantity (property): доступ к вычисленному количеству.
        - calculate_quantity(ta_text): пересчет количества по тексту.

    Логика вычисления количества (calculate_quantity):
        - Из текста извлекаются все числа, идущие после "TA"/"ТА"
          (regex: r"\\d*[ТT][АA]+[A-Za-z]?(\\d+)").
        - Если совпадений нет или числа не найдены — quantity = 0.
        - Если найдено одно число — quantity = 0 (нельзя вычислить диапазон).
        - Иначе quantity = max(numbers) - min(numbers).
    """

    __ta_name = ""
    __ta_quantity = 0

    def __init__(self, ta_name):
        """
        Инициализирует объект Trans_TA и сразу вычисляет количество
        трансформаторов по переданной маркировке.

        Args:
            ta_name (str): Текстовая маркировка трансформатора тока
                (например, "TA1, TA3" или "ТА1-ТА4").
        """
        self.__ta_name = ta_name
        self.calculate_quantity(ta_name)

    @property
    def ta_name(self):
        """
        Возвращает имя (маркировку) трансформатора тока.

        Returns:
            str: Текущее значение __ta_name.
        """
        return self.__ta_name

    @property
    def ta_quantity(self):
        """
        Возвращает вычисленное количество трансформаторов тока.

        Returns:
            int: Текущее значение __ta_quantity.
        """
        return self.__ta_quantity

    @ta_name.setter
    def ta_name(self, ta_name):
        """
        Устанавливает новое имя (маркировку) трансформатора тока.

        Важно: сеттер НЕ пересчитывает количество автоматически.
        Для пересчета нужно явно вызвать calculate_quantity(ta_name).

        Args:
            ta_name (str): Новое имя/маркировка.

        Returns:
            None
        """
        self.__ta_name = ta_name

    def calculate_quantity(self, ta_text):
        """
        Вычисляет количество трансформаторов тока по текстовой маркировке.

        Алгоритм:
            1. Если ta_text пустой/None — quantity = 0.
            2. Ищет в тексте все числа, идущие после "TA"/"ТА"
               (regex: r"\\d*[ТT][АA]+[A-Za-z]?(\\d+)").
            3. Если совпадений нет — quantity = 0.
            4. Преобразует найденные строки в int, отбрасывая нечисловые.
            5. Если чисел нет — quantity = 0.
            6. Если число одно — quantity = 0 (диапазон не определить).
            7. Иначе quantity = max(numbers) - min(numbers).

        При любом исключении логируется ошибка и quantity = 0.

        Args:
            ta_text (str): Текстовая маркировка трансформатора тока.

        Returns:
            None: Результат сохраняется в self.__ta_quantity.
        """
        if not ta_text:
            self.__ta_quantity = 0  # default значение
            return
        try:
            pattern = r"\d*[ТT][АA]+[A-Za-z]?(\d+)"
            matches = re.findall(pattern, ta_text)

            if not matches:
                self.__ta_quantity = 0
                return

            transformer_numbers = [int(num) for num in matches if num.isdigit()]

            if not transformer_numbers:
                self.__ta_quantity = 0
                return

            if len(transformer_numbers) == 1:
                self.__ta_quantity = 0
                return

            min_num = min(transformer_numbers)
            max_num = max(transformer_numbers)

            self.__ta_quantity = max_num - min_num

        except Exception as e:
            print(f"Error calculating TA quantity: {e}")
            self.__ta_quantity = 0


def create_ta(ta_text):
    """
    Фабричная функция для создания объекта Trans_TA.

    Args:
        ta_text (str): Текстовая маркировка трансформатора тока.

    Returns:
        Trans_TA: Созданный объект с уже вычисленным полем ta_quantity.
    """
    new_ta = Trans_TA(ta_text)
    return new_ta