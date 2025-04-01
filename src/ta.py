import re


class Trans_TA:
    __ta_name = ''
    __ta_quantity = 0

    def __init__(self, ta_name):
        self.__ta_name = ta_name


    @property
    def ta_name(self):
        return self.__ta_name

    @property
    def ta_quantity(self):
        return self.__ta_quantity

    @ta_name.setter
    def ta_name(self, ta_name):
        self.__ta_name = ta_name
    
    @ta_quantity.setter
    def ta_quantity(self, ta_text):
        pattern = r"\d*[ТT][АA]+[A-Za-z]?(\d+)"
        matches = re.findall(pattern, ta_text)
        transformer_numbers = [int(num) for num in matches]
        
        #в теории тут должно быть +1, но я думаю добавлять недостающее количество к числу найденных объектов на изображении
        self.__ta_quantity = max(transformer_numbers) - min(transformer_numbers)




def create_ta(ta_text):
    new_ta = Trans_TA(ta_text)
    new_ta.ta_quantity = new_ta.ta_name
    return new_ta