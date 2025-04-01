
class QF():
    __ID_QF = ''
    __Current = ''
    __Voltage = ''
    __Current_Close = ''

    def __init__(self, ID_QF, Current, Voltage, Current_Close):
        self.__ID_QF = ID_QF
        self.__Current = Current
        self.__Voltage = Voltage
        self.__Current_Close = Current_Close

    @property
    def ID_QF(self):
        return self.__ID_QF

    @property 
    def Current(self):
        return self.__Current
    
    @property
    def Voltage(self):
        return self.__Voltage
    
    @property
    def Current_Close(self):
        return self.__Current_Close
    
    @ID_QF.setter
    def ID_QF(self, ID_QF):
        self.__ID_QF = ID_QF

    @Current.setter
    def Current(self, Current):
        self.__Current = Current

    @Voltage.setter
    def Voltage(self, Voltage):
        self.__Voltage = Voltage

    @Current_Close.setter
    def Current_Close(self, Current_Close):
        self.__Current_Close = Current_Close    
  
    def print_data(self):
        print(self.__ID_QF, ':', self.__Current, self.__Voltage)


def create_qf(ID_QF, Current, Voltage, Current_Close):
    new_qf = QF(ID_QF, Current, Voltage, Current_Close)  # Создаем новый объект QF
    return new_qf

