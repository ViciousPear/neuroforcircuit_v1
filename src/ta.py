import re

class Trans_TA():
    __ta_name = ""
    __ta_quantity = 0

    def __init__(self, ta_name):
        self.__ta_name = ta_name
        self.calculate_quantity(ta_name)

    @property
    def ta_name(self):
        return self.__ta_name

    @property
    def ta_quantity(self):
        return self.__ta_quantity

    @ta_name.setter
    def ta_name(self, ta_name):
        self.__ta_name = ta_name
    
    def calculate_quantity(self, ta_text):
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
            
            #в теории тут должно быть +1, но я думаю добавлять недостающее количество к числу найденных объектов на изображении
            self.__ta_quantity = max_num - min_num

        except Exception as e:
            print(f"Error calculating TA quantity: {e}")
            self.__ta_quantity = 0




def create_ta(ta_text):
    new_ta = Trans_TA(ta_text)
    return new_ta