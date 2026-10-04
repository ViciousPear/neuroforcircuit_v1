import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tempfile
import os
from datetime import datetime
import webbrowser
from PIL import Image
import requests
import base64
import cv2
from src import search_object
from db_api import db_client_api
from src import qf


class DetectionApp:
    """
    GUI-приложение для анализа однолинейных схем NeuroElcom.

    Обеспечивает три режима работы:
        1. process_auto — обработка через внешний API
           (POST /detect_images/).
        2. process_with_edit_direct — локальное распознавание через
           search_object.recognize_only + диалог редактирования
           автоматов + поиск в БД (db_client).
        3. process_image — быстрый анализ через API клиент
           (self.api_client.detect_image).

    Также содержит кнопку открытия инструкции и статус-бар.

    Атрибуты:
        root (tk.Tk): Корневое окно Tk.
        api_client: Клиент API (может быть None — тогда режимы,
            требующие API, будут недоступны).
        search_object: Модуль search_object.
        db_client (DBClient): Клиент базы данных.
        qf: Модуль qf.
        temp_image_path (str | None): Путь к временному изображению
            с результатами.
        temp_text_path (str | None): Путь к временному текстовому
            отчету.
        current_image_data (str | None): Base64-изображение текущего
            результата (для сохранения после редактирования).
        detected_elements (list[dict]): Последние распознанные элементы.
    """

    def __init__(self, root, api_client=None):
        """
        Инициализирует приложение, загружает модули и создает виджеты.

        Args:
            root (tk.Tk): Корневое окно Tk.
            api_client: Клиент API (опционально). Если None — режимы,
                работающие через API, покажут сообщение об ошибке.

        Returns:
            None

        Raises:
            ImportError: Если не удалось загрузить search_object,
                db_client_api или qf (пробрасывается после
                messagebox.showerror).
        """
        self.root = root
        self.root.title("Анализатор однолинейных схем NeuroElcom")
        self.root.geometry("600x350")

        self.api_client = api_client

        self.initialize_modules()

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

        self.temp_image_path = None
        self.temp_text_path = None

        # Для хранения промежуточных данных
        self.current_image_data = None
        self.detected_elements = []

        self.create_widgets()

    def initialize_modules(self):
        """
        Загружает search_object, DBClient и qf, проверяет их наличие.

        Печатает в stdout результаты проверки (наличие recognize_only,
        наличие db_client, наличие qf). При ImportError показывает
        messagebox и пробрасывает исключение.

        Returns:
            None

        Raises:
            ImportError: Если какой-либо из модулей не загрузился.
        """
        try:
            self.search_object = search_object
            self.db_client = db_client_api.DBClient()
            self.qf = qf

            print("Модули успешно загружены:")
            print(f"  - search_object: {hasattr(self.search_object, 'recognize_only')}")
            print(f"  - db_client: {self.db_client is not None}")
            print(f"  - qf: {self.qf is not None}")

        except ImportError as e:
            print(f"Ошибка загрузки модулей: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить модули: {e}")
            raise

    def create_widgets(self):
        """
        Создает все виджеты главного окна: заголовок, четыре кнопки
        (Автоматический режим, Режим с редактированием, Быстрый анализ,
        Открыть инструкцию) и статус-бар.

        Returns:
            None
        """
        # Основная рамка
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(pady=30)

        # Заголовок
        title_label = tk.Label(
            main_frame,
            text="NeuroElcom - Анализ однолинейных схем",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0'
        )
        title_label.pack(pady=10)

        # Кнопка 1: Автоматический режим
        self.auto_btn = tk.Button(
            main_frame,
            text="Автоматический режим",
            command=self.process_auto,
            **self.button_style
        )
        self.auto_btn.pack(pady=8)

        # Кнопка 2: Режим с редактированием
        self.edit_btn = tk.Button(
            main_frame,
            text="Режим с редактированием",
            command=self.process_with_edit_direct,
            **self.button_style
        )
        self.edit_btn.pack(pady=8)

        # Кнопка 3: Быстрый анализ
        self.legacy_btn = tk.Button(
            main_frame,
            text="Быстрый анализ",
            command=self.process_image,
            **self.button_style
        )
        self.legacy_btn.pack(pady=8)

        # Кнопка 4: Инструкция
        self.instruction_btn = tk.Button(
            main_frame,
            text="Открыть инструкцию",
            command=self.open_instruction,
            **self.button_style
        )
        self.instruction_btn.pack(pady=8)

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

    def process_auto(self):
        """
        Автоматический режим: отправляет выбранное изображение на
        внешний API (POST {api_client.base_url}/detect_images/).

        Пайплайн:
            1. Проверяет, что api_client задан.
            2. Удаляет старые временные файлы (cleanup_old_files).
            3. Открывает диалог выбора изображения.
            4. Отправляет файл в API (multipart/form-data).
            5. Декодирует base64-изображение и сохраняет его
               (save_detection_image).
            6. Создает текстовый отчет (create_report_file) и
               открывает результаты (open_results).

        Returns:
            None

        Raises:
            Не пробрасывает исключения — ловит их и показывает
            messagebox.showerror.
        """
        if not self.api_client:
            messagebox.showerror("Ошибка", "API клиент не настроен")
            return

        self.cleanup_old_files()

        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        self.status_var.set("Обработка через API...")
        self.root.update()

        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = self.api_client.session.post(
                    f"{self.api_client.base_url}/detect_images/",
                    files=files,
                    timeout=600
                )

            if response.status_code != 200:
                raise Exception(f"Ошибка API: {response.text}")

            data = response.json()
            image_bytes = base64.b64decode(data["image_base64"])

            self.save_detection_image(image_bytes)
            self.create_report_file(data["detection_results"])
            self.open_results()

            self.status_var.set("Обработка завершена")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.status_var.set("Ошибка обработки")

    def process_with_edit_direct(self):
        """
        Режим с редактированием: локальное распознавание через
        search_object.recognize_only + диалог редактирования автоматов.

        Пайплайн:
            1. Удаляет старые временные файлы.
            2. Открывает диалог выбора изображения.
            3. Копирует файл во временный .jpg с латинским именем
               (temp_<uuid>.jpg) — на случай кириллицы в пути.
            4. Вызывает search_object.recognize_only(temp_file_path).
            5. Кодирует полученное изображение в base64 и сохраняет
               в self.current_image_data.
            6. Фильтрует элементы по типам QF/transformer/counter.
            7. Если QF есть — открывает EditDialogDirect, передавая
               callback on_edit_complete_direct.
            8. В finally удаляет временный файл.

        Returns:
            None

        Raises:
            Не пробрасывает исключения — ловит их и показывает
            messagebox.showerror.
        """
        self.cleanup_old_files()

        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        self.status_var.set("Локальное распознавание...")
        self.root.update()

        try:
            # Создание временного файла с латинским именем
            temp_dir = tempfile.gettempdir()
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            import uuid
            temp_filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
            temp_file_path = os.path.join(temp_dir, temp_filename)

            with open(temp_file_path, 'wb') as f:
                f.write(file_bytes)

            try:
                # 1. Распознавание
                print("Вызываем recognize_only...")
                image, detected_elements = self.search_object.recognize_only(temp_file_path)

                self.detected_elements = detected_elements

                _, img_encoded = cv2.imencode('.jpg', image)
                self.current_image_data = base64.b64encode(img_encoded).decode('utf-8')

                # Фильтрация QF элементов для редактирования
                qf_elements = [e for e in self.detected_elements if e.get("type") == "QF"]
                transformers = [e for e in self.detected_elements if e.get("type") == "transformer"]
                counters = [e for e in self.detected_elements if e.get("type") == "counter"]

                print(f"Распознано: {len(qf_elements)} автоматов, {len(transformers)} трансформаторов, {len(counters)} счетчиков")

                # 2. Показ диалога редактирования
                if qf_elements:
                    edit_dialog = EditDialogDirect(
                        self.root,
                        qf_elements,
                        transformers,
                        counters,
                        self.on_edit_complete_direct
                    )
                    self.status_var.set("Ожидание редактирования...")
                else:
                    messagebox.showinfo("Информация", "Автоматические выключатели не обнаружены")
                    self.status_var.set("Готов к работе")

            finally:
                # Удаление временного файл
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка распознавания: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Ошибка")

    def on_edit_complete_direct(self, edited_qf_elements, transformers, counters):
        """
        Callback после редактирования: выполняет поиск в БД по
        отредактированным QF, а также по трансформаторам и счетчикам.

        Пайплайн:
            1. Для каждого QF-элемента создает объект qf.QF
               (с фолбэком на QF.from_dict при ошибке) и вызывает
               db_client.search_breakers(qf_obj).
            2. Для каждого трансформатора вызывает
               db_client.search_transformators().
            3. Для каждого счетчика вызывает db_client.search_counters().
            4. Сохраняет изображение (если есть current_image_data).
            5. Формирует отчет (create_report_file) и открывает
               результаты (open_results).
            6. Показывает messagebox с итогами.

        Args:
            edited_qf_elements (list[dict]): Отредактированные QF-элементы.
            transformers (list[dict]): Трансформаторы (без изменений).
            counters (list[dict]): Счетчики (без изменений).

        Returns:
            None

        Raises:
            Не пробрасывает исключения — ловит их и показывает
            messagebox.showerror.
        """
        try:
            self.status_var.set("Поиск в БД...")
            self.root.update()

            print(f"Редактировано QF элементов: {len(edited_qf_elements)}")
            print(f"Трансформаторов: {len(transformers)}, Счетчиков: {len(counters)}")

            # 1. Поиск автоматов
            qf_results = []
            for element in edited_qf_elements:
                params = element["parameters"]

                id_qf = str(params.get("id_qf", "")) if params.get("id_qf") is not None else ""
                current = str(params.get("current", "")) if params.get("current") is not None else ""
                voltage = str(params.get("voltage", "")) if params.get("voltage") is not None else ""
                current_close = str(params.get("current_close", "")) if params.get("current_close") is not None else ""
                mounting_type = str(params.get("mounting_type", "")) if params.get("mounting_type") is not None else ""
                name = str(params.get("name", "")) if params.get("name") is not None else ""
                polus = str(params.get("polus", "")) if params.get("polus") is not None else ""

                print(f"Создаем QF объект с параметрами:")
                print(f"  - mounting_type (тип): {type(mounting_type)}, значение: '{mounting_type}'")

                try:
                    qf_obj = self.qf.QF(
                        ID_QF=id_qf,
                        Current=current,
                        Voltage=voltage,
                        Current_Close=current_close,
                        Mounting_Type=mounting_type,
                        Name=name,
                        Polus=polus
                    )
                    print(f"  - QF объект создан успешно")
                except Exception as e:
                    print(f"  - Ошибка создания QF объекта: {e}")
                    # Альтернативный способ создания
                    qf_obj = self.qf.QF.from_dict({
                        "ID_QF": id_qf,
                        "Current": current,
                        "Voltage": voltage,
                        "Current_Close": current_close,
                        "Mounting_Type": mounting_type,
                        "Name": name,
                        "Polus": polus
                    })
                    print(f"  - QF объект создан через from_dict")

                # Прямой вызов db_client
                try:
                    results = self.db_client.search_breakers(qf_obj)
                    qf_results.append(results)
                    print(f"  - Найдено автоматов для QF: {len(results)}")
                except Exception as e:
                    print(f"  - Ошибка поиска в БД: {e}")
                    qf_results.append([])

            # 2. Поиск трансформаторов
            transformer_results = []
            if transformers:
                print(f"Поиск {len(transformers)} трансформаторов...")
                for i in range(len(transformers)):
                    try:
                        results = self.db_client.search_transformators()
                        transformer_results.append(results)
                        print(f"  - Найдено трансформаторов: {len(results)}")
                    except Exception as e:
                        print(f"  - Ошибка поиска трансформаторов: {e}")
                        transformer_results.append([])

            # 3. Поиск счетчиков
            counter_results = []
            if counters:
                print(f"Поиск {len(counters)} счетчиков...")
                for i in range(len(counters)):
                    try:
                        results = self.db_client.search_counters()
                        counter_results.append(results)
                        print(f"  - Найдено счетчиков: {len(results)}")
                    except Exception as e:
                        print(f"  - Ошибка поиска счетчиков: {e}")
                        counter_results.append([])

            # Сохранение изображения
            if self.current_image_data:
                try:
                    image_bytes = base64.b64decode(self.current_image_data)
                    self.save_detection_image(image_bytes)
                except Exception as e:
                    print(f"Ошибка сохранения изображения: {e}")

            # Формирование отчетв
            all_results = [qf_results, transformer_results, counter_results]
            self.create_report_file(all_results)

            # Открытие результатов
            self.open_results()

            self.status_var.set("Обработка завершена")

            total_qf = sum(len(r) for r in qf_results) if qf_results else 0
            total_trans = sum(len(r) for r in transformer_results) if transformer_results else 0
            total_counters = sum(len(r) for r in counter_results) if counter_results else 0

            messagebox.showinfo("Успех",
                f"Поиск завершен:\n"
                f"• Автоматов: {len(qf_results)} позиций ({total_qf} вариантов)\n"
                f"• Трансформаторов: {len(transformer_results)} позиций ({total_trans} вариантов)\n"
                f"• Счетчиков: {len(counter_results)} позиций ({total_counters} вариантов)")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка поиска в БД: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Ошибка")

    def detect_image(self, image_path):
        """
        Обертка над self.api_client.detect_image(image_path).

        Args:
            image_path (str): Путь к изображению.

        Returns:
            tuple: (image_bytes, detection_results) — то, что вернет
                api_client.detect_image.

        Raises:
            Exception: При requests.exceptions.RequestException —
                оборачивается в Exception с текстом "Connection error: ...".
        """
        try:
            return self.api_client.detect_image(image_path)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {str(e)}")

    def cleanup_old_files(self):
        """
        Удаляет старые временные файлы (изображение и отчет), если они
        существуют, и сбрасывает self.temp_image_path / self.temp_text_path.

        Returns:
            None
        """
        if not hasattr(self, 'temp_image_path'):
            self.temp_image_path = None
        if not hasattr(self, 'temp_text_path'):
            self.temp_text_path = None

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
        """
        Открывает PDF-инструкцию в браузере по умолчанию.

        Returns:
            None
        """
        webbrowser.open("Инструкция_по_использованию_приложения.pdf")

    def process_image(self):
        """
        Быстрый анализ: отправляет изображение через self.api_client.detect_image
        и сохраняет/открывает результаты.

        Пайплайн:
            1. cleanup_old_files.
            2. Диалог выбора изображения.
            3. detect_image(file_path) → (image_bytes, detection_results).
            4. save_detection_image, create_report_file, open_results.

        Returns:
            None

        Raises:
            Не пробрасывает исключения — ловит их и показывает
            messagebox.showerror.
        """
        self.cleanup_old_files()

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
            image_bytes, detection_results = self.detect_image(file_path)

            self.save_detection_image(image_bytes)
            self.create_report_file(detection_results)
            self.open_results()

            self.status_var.set("Обработка завершена")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            self.status_var.set("Ошибка обработки")

    def save_detection_image(self, image_bytes):
        """
        Сохраняет изображение с результатами во временную директорию.

        Имя файла: detection_result_<timestamp>.jpg.

        Args:
            image_bytes (bytes): Байты изображения (JPEG).

        Returns:
            None
        """
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_image_path = os.path.join(temp_dir, f"detection_result_{timestamp}.jpg")

        with open(self.temp_image_path, 'wb') as f:
            f.write(image_bytes)

    def create_report_file(self, results):
        """
        Формирует текстовый отчет по результатам поиска в БД.

        Отчет содержит три раздела: автоматические выключатели,
        трансформаторы, счетчики. Для каждой группы (позиции)
        выводятся найденные артикулы с названием и стоимостью.
        В конце — итоговое количество найденных позиций.

        Args:
            results (list): Список из трех элементов:
                [qf_results, transformer_results, counter_results],
                где каждый — список групп (позиций), а группа —
                список dict-элементов с полями 'id', 'name', 'price'.

        Returns:
            None: Отчет сохраняется в self.temp_text_path.
        """
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_text_path = os.path.join(temp_dir, f"detection_report_{timestamp}.txt")

        report_lines = []

        report_lines.append("="*60)
        report_lines.append("ОТЧЕТ ОБ ОБНАРУЖЕННОМ ОБОРУДОВАНИИ")
        report_lines.append("="*60)
        report_lines.append(f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        report_lines.append(f"Режим работы: С редактированием параметров")
        report_lines.append("")

        section_names = {
            0: "АВТОМАТИЧЕСКИЕ ВЫКЛЮЧАТЕЛИ",
            1: "ТРАНСФОРМАТОРЫ",
            2: "СЧЕТЧИКИ"
        }

        total_items_found = 0

        for section_idx, section_data in enumerate(results):
            section_name = section_names.get(section_idx, f"Раздел {section_idx+1}")
            report_lines.append(f"{section_name}:")
            report_lines.append("-"*60)

            if not section_data or not isinstance(section_data, list):
                report_lines.append("Не обнаружено подходящего оборудования\n")
                continue

            section_items_found = 0

            for group_idx, group in enumerate(section_data, 1):
                if not group or not isinstance(group, list):
                    continue

                if len(group) > 0:
                    report_lines.append(f"\nПозиция {group_idx}:")
                    report_lines.append("-"*40)

                    section_items_found += 1
                    for item_idx, item in enumerate(group, 1):
                        if isinstance(item, dict):
                            report_lines.append(
                                f"{item_idx}. Артикул: {item.get('id', 'N/A')}\n"
                                f"   Название: {item.get('name', 'N/A')}\n"
                                f"   Стоимость: {item.get('price', 'N/A')} руб.\n"
                            )
                        else:
                            report_lines.append(f"{item_idx}. Неверный формат данных")

            if section_items_found > 0:
                report_lines.append(f"\nВсего позиций в разделе: {section_items_found}")
                total_items_found += section_items_found
            else:
                report_lines.append("Не обнаружено подходящего оборудования")

            report_lines.append("\n")

        report_lines.append("="*60)
        report_lines.append("ИТОГО:")
        report_lines.append(f"Всего найдено позиций оборудования: {total_items_found}")

        with open(self.temp_text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"Отчет сохранен: {self.temp_text_path}")

    def open_results(self):
        """
        Открывает сохраненное изображение (через PIL.Image.show) и
        текстовый отчет (через webbrowser.open).

        Returns:
            None
        """
        if self.temp_image_path and os.path.exists(self.temp_image_path):
            img = Image.open(self.temp_image_path)
            img.show()

        if self.temp_text_path and os.path.exists(self.temp_text_path):
            webbrowser.open(self.temp_text_path)

    def __del__(self):
        """
        Деструктор: при сборке мусора удаляет временные файлы.

        Returns:
            None
        """
        self.cleanup_old_files()


class EditDialogDirect:
    """
    Окно редактирования параметров автоматических выключателей
    с последующим поиском в БД (без промежуточного API).

    Позволяет редактировать поля Name, Current, Voltage, Current_Close,
    Polus, Mounting_Type для каждого QF-элемента. Трансформаторы и
    счетчики передаются в callback без изменений.

    Атрибуты:
        parent (tk.Tk | tk.Toplevel): Родительское окно.
        qf_elements (list[dict]): QF-элементы для редактирования.
        transformers (list[dict]): Трансформаторы.
        counters (list[dict]): Счетчики.
        callback (callable): Функция, вызываемая при сохранении:
            callback(edited_qf_elements, transformers, counters).
        dialog (tk.Toplevel): Окно диалога.
        tree (ttk.Treeview): Таблица редактирования.
        editing_cell (tuple | None): Текущая редактируемая ячейка
            (row_id, column_index, entry).
        edit_window (tk.Toplevel | None): Всплывающее окно ввода.
    """

    def __init__(self, parent, qf_elements, transformers, counters, callback):
        """
        Создает диалог редактирования.

        Args:
            parent: Родительское окно.
            qf_elements (list[dict]): QF-элементы.
            transformers (list[dict]): Трансформаторы.
            counters (list[dict]): Счетчики.
            callback (callable): Функция, вызываемая при сохранении.

        Returns:
            None
        """
        self.parent = parent
        self.qf_elements = qf_elements
        self.transformers = transformers
        self.counters = counters
        self.callback = callback

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Редактирование параметров")
        self.dialog.geometry("850x600")

        # Заголовок
        title_label = tk.Label(
            self.dialog,
            text="Редактирование автоматических выключателей",
            font=('Arial', 12, 'bold')
        )
        title_label.pack(pady=10)

        # Информация о других элементах
        info_text = (
            f"Обнаружено на схеме:\n"
            f"• Автоматических выключателей: {len(qf_elements)}\n"
            f"• Трансформаторов: {len(transformers)}\n"
            f"• Счетчиков: {len(counters)}\n\n"
            f"Редактируйте параметры автоматов ниже. Трансформаторы и счетчики\n"
            f"будут автоматически переданы для поиска в БД."
        )
        info_label = tk.Label(
            self.dialog,
            text=info_text,
            font=('Arial', 10),
            justify=tk.LEFT,
            bg='#f0f8ff',
            relief=tk.RIDGE,
            padx=10,
            pady=10
        )
        info_label.pack(pady=5, padx=10, fill=tk.X)

        # Таблица
        columns = ("№", "Имя", "Ток (А)", "Напряжение (В)", "Откл. спос.", "Полюсы", "Монтаж")
        self.tree = ttk.Treeview(self.dialog, columns=columns, show="headings", height=10)

        column_widths = [50, 120, 80, 100, 100, 80, 100]
        for idx, col in enumerate(columns):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[idx], minwidth=50)

        scrollbar = ttk.Scrollbar(self.dialog, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

        # Заполнение таблицы
        self.load_elements()

        # Инструкция
        instruction = tk.Label(
            self.dialog,
            text="Дважды кликните по ячейке для редактирования",
            font=('Arial', 9),
            fg='#666'
        )
        instruction.pack(pady=5)

        # Кнопки
        btn_frame = tk.Frame(self.dialog)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Сохранить и искать в БД",
                 command=self.save_and_search, bg="#4CAF50", fg="white",
                 font=('Arial', 10), width=20, height=1).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Отмена",
                 command=self.dialog.destroy, width=15, height=1).pack(side=tk.LEFT, padx=5)

        # Настройка редактирования
        self.setup_cell_editing()

    def load_elements(self):
        """
        Загружает QF-элементы в таблицу Treeview.

        Для каждого элемента берет element["parameters"] и заполняет
        колонки: номер, имя, ток, напряжение, ток отключения, полюсы,
        тип монтажа. В tags сохраняется element["id"].

        Returns:
            None
        """
        for element in self.qf_elements:
            params = element["parameters"]
            element_number = element.get("number", len(self.tree.get_children()) + 1)
            values = (
                element_number,
                params.get("name", ""),
                params.get("current", ""),
                params.get("voltage", ""),
                params.get("current_close", ""),
                params.get("polus", ""),
                params.get("mounting_type", "")
            )
            self.tree.insert("", tk.END, values=values, tags=(element["id"],))

    def setup_cell_editing(self):
        """
        Настраивает обработку двойного клика по ячейкам таблицы.

        Returns:
            None
        """
        self.editing_cell = None
        self.edit_window = None
        self.tree.bind("<Double-1>", self.on_double_click)

    def on_double_click(self, event):
        """
        Обработчик двойного клика: определяет ячейку и открывает
        всплывающее окно редактирования.

        Колонка № (индекс 0) не редактируется.

        Args:
            event: Событие Tk.

        Returns:
            None
        """
        region = self.tree.identify_region(event.x, event.y)
        if region == "cell":
            column = self.tree.identify_column(event.x)
            row_id = self.tree.identify_row(event.y)

            item = self.tree.item(row_id)
            column_index = int(column[1:]) - 1
            current_value = item["values"][column_index]

            if column_index == 0:  # Пропускаем номер
                return

            self.create_edit_window(row_id, column_index, current_value, event.x, event.y)

    def create_edit_window(self, row_id, column_index, current_value, x, y):
        """
        Создает всплывающее окно Entry для редактирования ячейки.

        Args:
            row_id (str): Идентификатор строки Treeview.
            column_index (int): Индекс колонки.
            current_value: Текущее значение ячейки.
            x (int): X-координата клика.
            y (int): Y-координата клика.

        Returns:
            None
        """
        if self.edit_window:
            self.edit_window.destroy()

        self.edit_window = tk.Toplevel(self.dialog)
        self.edit_window.wm_overrideredirect(True)
        self.edit_window.wm_geometry(f"+{x+self.dialog.winfo_rootx()}+{y+self.dialog.winfo_rooty()}")

        entry = tk.Entry(self.edit_window)
        entry.insert(0, current_value)
        entry.pack()
        entry.focus_set()

        self.editing_cell = (row_id, column_index, entry)

        entry.bind("<Return>", lambda e: self.save_cell_edit())
        entry.bind("<Escape>", lambda e: self.cancel_cell_edit())
        entry.bind("<FocusOut>", lambda e: self.save_cell_edit())

    def save_cell_edit(self):
        """
        Сохраняет отредактированное значение ячейки в Treeview и
        закрывает всплывающее окно.

        Returns:
            None
        """
        if not self.editing_cell:
            return

        row_id, column_index, entry = self.editing_cell
        new_value = entry.get()

        item = self.tree.item(row_id)
        values = list(item["values"])
        values[column_index] = new_value
        self.tree.item(row_id, values=values)

        if self.edit_window:
            self.edit_window.destroy()
            self.edit_window = None
            self.editing_cell = None

    def cancel_cell_edit(self):
        """
        Отменяет редактирование ячейки: закрывает всплывающее окно
        без сохранения.

        Returns:
            None
        """
        if self.edit_window:
            self.edit_window.destroy()
            self.edit_window = None
            self.editing_cell = None

    def save_and_search(self):
        """
        Собирает отредактированные значения из таблицы, формирует
        edited_elements и вызывает callback(edited_elements,
        transformers, counters). Закрывает диалог.

        Returns:
            None
        """
        if self.edit_window:
            self.edit_window.destroy()

        edited_elements = []

        for item_id in self.tree.get_children():
            values = self.tree.item(item_id)["values"]
            original_id = self.tree.item(item_id)["tags"][0] if self.tree.item(item_id)["tags"] else ""

            original_element = None
            for elem in self.qf_elements:
                if elem["id"] == original_id:
                    original_element = elem
                    break

            if original_element:
                edited_element = original_element.copy()
                edited_element["parameters"]["name"] = values[1]
                edited_element["parameters"]["current"] = values[2]
                edited_element["parameters"]["voltage"] = values[3]
                edited_element["parameters"]["current_close"] = values[4]
                edited_element["parameters"]["polus"] = values[5]
                edited_element["parameters"]["mounting_type"] = values[6]

                edited_elements.append(edited_element)

        self.dialog.destroy()
        self.callback(edited_elements, self.transformers, self.counters)


def main(api_client=None):
    """
    Точка входа приложения: создает Tk-окно, экземпляр DetectionApp
    и запускает mainloop.

    Args:
        api_client: Клиент API (опционально). Если None — режимы,
            требующие API, будут недоступны.

    Returns:
        None
    """
    root = tk.Tk()
    app = DetectionApp(root, api_client=api_client)
    root.mainloop()


if __name__ == "__main__":
    main()