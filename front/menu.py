import tkinter as tk
from tkinter import filedialog, messagebox
import tempfile
import os
from datetime import datetime
import webbrowser
from PIL import Image
import requests
import base64


class DetectionApp:
    def __init__(self, root, api_client=None):
        self.root = root
        self.root.title("Анализатор однолинейных схем NeuroElcom")
        self.root.geometry("600x250")
        
        self.api_client = api_client
        
        # Стилизация
        self.root.configure(bg='#f0f0f0')
        self.button_style = {
            'font': ('Arial', 12),
            'bg': '#4CAF50',
            'fg': 'white',
            'activebackground': '#45a049',
            'width': 25,
            'height': 2,
            'borderwidth': 2,
            'highlightthickness': 0
        }
        
        self.create_widgets()
        
        # Временные файлы
        self.temp_image_path = None
        self.temp_text_path = None
    
    def create_widgets(self):
        # Основная рамка
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(pady=40)
        
        # Кнопка выбора изображения
        self.select_btn = tk.Button(
            main_frame,
            text="Выбрать изображение",
            command=self.process_image,
            **self.button_style
        )
        self.select_btn.pack(pady=10)
        
        self.instruction_btn = tk.Button(
            main_frame,
            text="Открыть инструкцию",
            command=self.open_instruction,
            **self.button_style
        )
        self.instruction_btn.pack(pady=10)
        
        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе")
        
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#e0e0e0',
            font=('Arial', 10)
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def cleanup_old_files(self):
        """Удаляет старые временные файлы перед новым анализом"""
        if self.temp_image_path and os.path.exists(self.temp_image_path):
            try:
                os.remove(self.temp_image_path)
            except Exception as e:
                print(f"Ошибка удаления изображения: {e}")
        
        if self.temp_text_path and os.path.exists(self.temp_text_path):
            try:
                os.remove(self.temp_text_path)
            except Exception as e:
                print(f"Ошибка удаления отчета: {e}")
        
        self.temp_image_path = None
        self.temp_text_path = None
    
    def open_instruction(self):
        webbrowser.open("Инструкция_по_использованию_приложения.pdf")

    def process_image(self):
        # Очищаем старые файлы перед новым анализом
        self.cleanup_old_files()
        
        # Выбор файла изображения
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.status_var.set("Обработка изображения...")
        self.root.update()
        
        try:
            #print(self.api_client.detect_image(file_path))
            image_bytes, detection_results = self.api_client.detect_image(file_path)
            
            # Сохраняем обработанное изображение
            self.save_detection_image(image_bytes)
            
            # Создаем текстовый отчет
            self.create_report_file(detection_results)
            
            # Открываем результаты
            self.open_results()
            
            self.status_var.set("Обработка завершена")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            self.status_var.set("Ошибка обработки")
    
    def save_detection_image(self, image_bytes):
        """Сохраняет обработанное изображение во временный файл"""
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_image_path = os.path.join(temp_dir, f"detection_result_{timestamp}.jpg")
        
        with open(self.temp_image_path, 'wb') as f:
            f.write(image_bytes)
    
    def create_report_file(self, results):
        """Создает текстовый отчет с результатами"""
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_text_path = os.path.join(temp_dir, f"detection_report_{timestamp}.txt")
        
        report_lines = []
        
        # Добавляем заголовок
        report_lines.append("="*50)
        report_lines.append("ОТЧЕТ ОБ ОБНАРУЖЕННОМ ОБОРУДОВАНИИ")
        report_lines.append("="*50)
        report_lines.append(f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        report_lines.append("\n")
        
        # Названия разделов
        section_names = {
            0: "АВТОМАТИЧЕСКИЕ ВЫКЛЮЧАТЕЛИ",
            1: "ТРАНСФОРМАТОРЫ",
            2: "СЧЕТЧИКИ"
        }
        
        # Обрабатываем каждый раздел
        for section_idx, section_data in enumerate(results):
            section_name = section_names.get(section_idx, f"Раздел {section_idx+1}")
            report_lines.append(f"{section_name}:")
            report_lines.append("-"*50)
            
            if not section_data or not isinstance(section_data, list):
                report_lines.append("Не обнаружено подходящего оборудования\n")
                continue
                
            quantity_positions = 0
            
            for group_idx, group in enumerate(section_data, 1):
                if not group or not isinstance(group, list):
                    continue
                    
                report_lines.append(f"\nПредлагаемые варианты по группе {group_idx} (позиции {1}-{len(group)}):")
                
                quantity_positions += 1
                for item_idx, item in enumerate(group, 1):
                    if isinstance(item, dict):
                        report_lines.append(
                            f"{item_idx}. Артикул: {item.get('id', 'N/A')}\n"
                            f"Название: {item.get('name', 'N/A')}\n"
                            f"Стоимость: {item.get('price', 'N/A')} руб.\n"
                        )
                    else:
                        report_lines.append(f"{item_idx}. Неверный формат данных")
                
            report_lines.append(f"\nПредполагаемое количество позиций: {quantity_positions}")
            report_lines.append("\n")
        
        # Сохраняем отчет в файл
        with open(self.temp_text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

    def open_results(self):
        """Открывает результаты в соответствующих программах"""
        # Открываем изображение
        if self.temp_image_path and os.path.exists(self.temp_image_path):
            img = Image.open(self.temp_image_path)
            img.show()
        
        # Открываем текстовый отчет
        if self.temp_text_path and os.path.exists(self.temp_text_path):
            webbrowser.open(self.temp_text_path)
    
    def __del__(self):
        """Удаляет временные файлы при закрытии приложения"""
        self.cleanup_old_files()


def main(api_client=None):
    root = tk.Tk()
    app = DetectionApp(root, api_client=api_client)
    root.mainloop()
    

if __name__ == "__main__":
    main()