import search_text
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import cv2
import os
import qf, ta


def send_to_yolo_service(image, yolo_url):
    """
    Отправляет изображение в YOLO-сервис для детекции объектов.

    Изображение кодируется в PNG и отправляется POST-запросом на
    {yolo_url}/detect_raw как multipart/form-data. Ожидается JSON-ответ
    с полем 'success' == True и данными детекции.

    Args:
        image (np.ndarray): Входное изображение (OpenCV BGR).
        yolo_url (str): Базовый URL YOLO-сервиса
            (например, "http://yolo-service:5001").

    Returns:
        dict: JSON-ответ YOLO-сервиса с результатами детекции
            (ключи 'results', 'names' и т.п.).

    Raises:
        ValueError: Если сервис вернул success=False.
        requests.exceptions.RequestException: При сетевой ошибке или
            неуспешном HTTP-статусе. Исключение логируется и
            пробрасывается дальше.
    """
    _, img_encoded = cv2.imencode('.png', image)
    files = {'file': ('image.png', img_encoded.tobytes(), 'image/png')}
    try:
        response = requests.post(
            f"{yolo_url}/detect_raw",
            files=files,
            timeout=600
        )
        response.raise_for_status()
        data = response.json()

        if not data.get('success'):
            raise ValueError("YOLO service returned error")

        return data

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при отправке в YOLO-сервис: {e}")
        raise


def recognize_with_text_service(connected_text_chunks, image, text_boxes, text_numbers, text_service_url=None):
    """
    Отправляет связанные текстовые чанки на распознавание в text-service
    с сохранением уже имеющейся нумерации.

    Пайплайн:
        1. Определяет URL text-service (аргумент → env TEXT_SERVICE_URL →
           значение по умолчанию).
        2. Настраивает requests.Session с retry-стратегией
           (total=3, backoff_factor=0.1, коды 429/500/502/503/504).
        3. Для каждого чанка вырезает bbox из изображения и распознает
           текст через search_text.recognize_text_from_bbox.
        4. Чанки с успешно распознанным текстом собираются в
           chunks_with_text и отправляются батчем POST-запросом на
           {text_service_url}/api/process-text-batch вместе с размерами
           изображения.
        5. Разбирает ответ сервиса и формирует:
            - list_for_searching: объекты QF (автоматы) с параметрами
              и типом монтажа из чанка;
            - list_for_quantity: объекты Trans_TA (трансформаторы тока)
              с вычисленным количеством.

    Ответ сервиса обрабатывается «безопасно»:
        - пропускаются элементы, не являющиеся dict;
        - поле "object" может быть null — заменяется на {};
        - тип элемента определяется по chunk["connected_element_type"],
          а при его отсутствии — по result["type"]
          ("circuit_breaker" / "current_transformer");
        - для трансформаторов имя TA берется из
          ta_dict["_Trans_TA__ta_name"], затем из первой строки
          распознанного текста, затем — как "TA_<номер текста>".

    Args:
        connected_text_chunks (list[dict]): Список чанков, каждый из
            которых содержит как минимум:
                - "bbox": [x1, y1, x2, y2]
                - "automatic_number": номер связанного элемента
                - "text_number": номер текста
                - "connected_element_type": тип связанного элемента
                - "mounting_type": тип монтажа (для автоматов)
        image (np.ndarray): Исходное изображение (OpenCV BGR).
        text_boxes (list): Список всех текстовых bbox (для справки/совместимости).
        text_numbers (dict): Карта {text_bbox: номер_текста}
            (для справки/совместимости).
        text_service_url (str | None): Базовый URL text-service. Если None —
            берется из переменной окружения TEXT_SERVICE_URL или
            значение по умолчанию "http://text-processing-service:5002".

    Returns:
        tuple[list, list]:
            - list_for_searching: список объектов QF (автоматические
              выключатели) для последующего поиска в БД.
            - list_for_quantity: список объектов Trans_TA
              (трансформаторы тока) с вычисленным количеством.

    Notes:
        - При отсутствии connected_text_chunks возвращает ([], []).
        - При ошибке в процессе обработки логирует исключение и
          возвращает уже накопленные результаты (возможно, пустые).
    """
    list_for_searching = []
    list_for_quantity = []

    if text_service_url is None:
        text_service_url = os.getenv("TEXT_SERVICE_URL", "http://text-processing-service:5002")

    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if connected_text_chunks:
        try:
            print(f"Отправление {len(connected_text_chunks)} связанных текстов в text-service...")

            chunks_with_text = []
            for chunk in connected_text_chunks:
                x1, y1, x2, y2 = chunk["bbox"]
                text_content = search_text.recognize_text_from_bbox(image, x1, y1, x2, y2)

                if text_content:
                    chunk_with_text = chunk.copy()
                    chunk_with_text["text"] = text_content
                    chunks_with_text.append(chunk_with_text)

                    auto_number = chunk.get("automatic_number", "")
                    text_number = chunk.get("text_number", "")
                    element_type = chunk.get("connected_element_type", "элемент")
                    print(f"{element_type} {auto_number} (Текст {text_number}): распознан текст '{text_content[:50]}...'")
                else:
                    print(f"Не удалось распознать текст для bbox {chunk['bbox']}")

            if not chunks_with_text:
                print("Нет распознанных текстов для отправки")
                return list_for_searching, list_for_quantity

            response = session.post(
                f"{text_service_url}/api/process-text-batch",
                json={
                    "chunks": chunks_with_text,
                    "image_size": [image.shape[1], image.shape[0]],
                    "image_shape": list(image.shape)
                },
                timeout=300
            )
            response.raise_for_status()

            batch_results = response.json()

            for i, result in enumerate(batch_results):
                if not isinstance(result, dict):
                    continue

                chunk = chunks_with_text[i]
                text_content = chunk.get("text", "")

                element_type = str(chunk.get("connected_element_type", "")).lower().strip()
                res_type = result.get("type")

                # 1. Сценарий: АВТОМАТЫ
                if element_type in ["automatic", "автомат"] or (not element_type and res_type == "circuit_breaker"):
                    qf_dict = result.get("object") or {}
                    if not isinstance(qf_dict, dict):
                        qf_dict = {}

                    mounting_type = chunk.get("mounting_type", "0")

                    qf_obj = qf.create_qf(
                        qf_dict.get("ID_QF", ""),
                        qf_dict.get("Name", ""),
                        qf_dict.get("Current", ""),
                        qf_dict.get("Voltage", ""),
                        qf_dict.get("Current_Close", ""),
                        qf_dict.get("Polus", ""),
                        mounting_type
                    )
                    list_for_searching.append(qf_obj)

                    automatic_number = chunk.get("automatic_number", "Unknown")
                    text_number = chunk.get("text_number", "Unknown")
                    qf_id = qf_dict.get("ID_QF") or "Unknown"
                    print(f"Автомат {automatic_number} (Текст {text_number}, монтаж {mounting_type}): '{text_content[:30]}' → {qf_id}")

                # 2. Сценарий: ТРАНСФОРМАТОРЫ
                elif element_type in ["transformer", "трансформатор"] or (not element_type and res_type == "current_transformer"):
                    ta_dict = result.get("object") or {}
                    if not isinstance(ta_dict, dict):
                        ta_dict = {}

                    ta_name = ta_dict.get("_Trans_TA__ta_name", "")

                    if not ta_name and text_content:
                        ta_name = text_content.split('\n')[0].strip()

                    if not ta_name:
                        ta_name = f"TA_{chunk.get('text_number', 'Unknown')}"

                    ta_obj = ta.create_ta(ta_name)
                    list_for_quantity.append(ta_obj)
                    print(f"Трансформатор: '{text_content[:30]}' → {ta_name}")

                else:
                    print(f"Пропущен текст с нетипичным элементом '{element_type}' (тип ответа сервиса: '{res_type}'): '{text_content[:30]}'")

        except Exception as e:
            print(f"Ошибка при обработке текстов: {e}")

    else:
        print("Нет связанных текстов для обработки")

    return list_for_searching, list_for_quantity