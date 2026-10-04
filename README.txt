
## Компоненты

### Микросервисы

| Сервис | Папка | Назначение |
|---|---|---|
| **YOLO API** | `docker/api/` | Детекция элементов схемы (bbox + классы) |
| **Super-Image API** | `super_image_api/` | Улучшение качества изображений (EDSR) |
| **PaddleOCR API** | `paddleocr/` | Распознавание текста на ROI |
| **Text Processing API** | `text_processing_api/` | Парсинг и нормализация распознанного текста |
| **DB API** | `db_api/` | Поиск оборудования в базе данных |
| **OCR-воркеры** | `containers/ml/` | Celery-воркеры для асинхронной OCR-обработки |

### Клиентские модули (`src`)

| Модуль | Назначение |
|---|---|
| `search_object.py` | Пайплайн распознавания: YOLO → граф → OCR → text-service |
| `circuit_graph.py` | Граф схемы: узлы (элементы/тексты), рёбра, адаптивные связи |
| `geometry.py` | Геометрия: перекрытия bbox, тип монтажа, расстояния |
| `visualization.py` | Отрисовка bbox, номеров и связей на изображении |
| `qf.py` | Pydantic-модель автоматического выключателя (QF) |
| `ta.py` | Модель трансформатора тока (Trans_TA) |
| `processing_text.py` | Извлечение QF/TA из текста, нормализация токов и диапазонов |
| `search_text.py` | OCR-клиент: `recognize_text_from_bbox`, `enhance_image`, `cleaned_image` |
| `services_client.py` | Клиенты YOLO и text-service |
| `graph_mapper.py` | Построение графа и карт соответствий текст ↔ элемент |
| `super_image_fixed.py` | Обёртка над EDSR (super-image) для улучшения изображений |

### База данных (`database`, `db_api`)

- **`database/`** — модули подключения и низкоуровневого взаимодействия с БД.
- **`db_api/`** — FastAPI-сервис и клиент (`db_client_api.DBClient`) с методами:
  - `search_breakers(qf_obj)` — поиск автоматических выключателей;
  - `search_transformators()` — поиск трансформаторов;
  - `search_counters()` — поиск счётчиков.

### Клиентские приложения (`front`, `front_api`)

- **`front/`** — тестовое приложение на tkinter для локальной отладки
  (класс `DetectionApp`, диалог `EditDialogDirect`).
- **`front_api/`** — FastAPI-сервис для взаимодействия клиента и системы:
  - `POST /detect_only/` — детекция без поиска в БД;
  - `GET /health` — проверка здоровья.

### Модели YOLO (`runs`)

- **`runs/`** — веса обученной модели YOLO и метаданные
  (`best.pt`, `data.yaml`, логи обучения).
- Модель используется YOLO API (`docker/api/`) для детекции элементов схемы.

---

## Установка

```bash
git clone https://github.com/<your-org>/neuroelcom.git
cd neuroelcom

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt