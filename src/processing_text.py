import re
import qf, ta
from typing import Set, Tuple
# from . import qf - для локального теста
# from . import ta - для локального теста


MAX_ALLOWED_CURRENT = 6300
"""
Максимально допустимый рабочий ток (А). Значения выше считаются
артефактами OCR и обрабатываются отдельно (усечение лишних цифр,
разбор слипшихся диапазонов).
"""

# Стандартные ряды номинальных токов (IEC / ГОСТ)
STANDARD_CURRENTS_SET: Set[float] = {
    1, 2, 3, 4, 5, 6, 8, 10, 12, 12.5, 12.8, 16, 20, 25, 32, 40, 50, 63, 80, 100,
    112, 125, 128, 140, 160, 189, 200, 221, 250, 252, 300, 315, 320, 350, 397,
    400, 500, 504, 560, 630, 640, 800, 1000, 1250, 1600, 2000, 2500, 3150, 3200,
    4000, 5000, 6300
}
"""
Множество стандартных номинальных токов (А) по IEC / ГОСТ.
Используется для проверки типовых значений и допуска +-10%.
"""

STANDARD_CURRENTS_LIST = sorted(list(STANDARD_CURRENTS_SET))
"""
Отсортированный список стандартных номинальных токов (А).
Используется для поиска ближайшего стандартного значения.
"""

# База уникальных известных диапазонов из базы данных: (мин_ток, макс_ток, формализованный_текст)
STANDARD_RANGES_DB: Tuple[Tuple[float, float, str], ...] = (
    (12, 32, "12-32А"), (12.5, 16, "12.5-16А"), (12.5, 32, "12.5-32А"), (12.8, 16, "12.8-16А"),
    (16, 20, "16-20А"), (16, 40, "16-40А"), (20, 25, "20-25А"), (25, 32, "25-32А"),
    (25, 63, "25-63А"), (32, 40, "32-40А"), (40, 50, "40-50А"), (40, 100, "40-100А"),
    (44, 63, "44-63А"), (50, 63, "50-63А"), (50, 125, "50-125А"), (50.4, 63, "50.4-63А"),
    (56, 80, "56-80А"), (60, 75, "60-75А"), (63, 80, "63-80А"), (63, 160, "63-160А"),
    (64, 80, "64-80А"), (64, 160, "64-160А"), (80, 100, "80-100А"), (87.5, 125, "87.5-125А"),
    (100, 125, "100-125А"), (100, 250, "100-250А"), (112, 160, "112-160А"), (125, 160, "125-160А"),
    (128, 160, "128-160А"), (140, 200, "140-200А"), (160, 200, "160-200А"), (160, 400, "160-400А"),
    (189, 300, "189-300А"), (200, 250, "200-250А"), (221, 350, "221-350А"), (250, 630, "250-630А"),
    (252, 315, "252-315А"), (252, 400, "252-400А"), (252, 630, "252-630А"), (315, 800, "315-800А"),
    (320, 400, "320-400А"), (320, 800, "320-800А"), (397, 630, "397-630А"), (400, 500, "400-500А"),
    (400, 1000, "400-1000А"), (500, 630, "500-630А"), (500, 1250, "500-1250А"), (504, 630, "504-630А"),
    (504, 800, "504-800А"), (630, 1600, "630-1600А"), (640, 800, "640-800А"), (640, 1600, "640-1600А"),
    (800, 1000, "800-1000А"), (1000, 1250, "1000-1250А")
)
"""
База известных стандартных диапазонов токов из БД:
(мин_ток, макс_ток, формализованный_текст).
Используется для точного и скорингового сопоставления.
"""

# Быстрый индекс уникальных пар (min, max) для проверки O(1)
EXACT_RANGES_MAP = {(item[0], item[1]): item[2] for item in STANDARD_RANGES_DB}
"""
Индекс { (min, max): "min-maxА" } для быстрого поиска точного
совпадения диапазона за O(1).
"""


def is_typical_current(value: float) -> bool:
    """
    Проверяет, является ли номинал типовым.

    Логика:
        - Если value точно входит в STANDARD_CURRENTS_SET — True.
        - Если value > MAX_ALLOWED_CURRENT или value < 1 — False.
        - Иначе — True, если value отличается от любого элемента
          STANDARD_CURRENTS_LIST не более чем на 10%.

    Args:
        value (float): Проверяемое значение тока (А).

    Returns:
        bool: True, если значение типовое, иначе False.
    """
    if value in STANDARD_CURRENTS_SET:
        return True

    if value > MAX_ALLOWED_CURRENT or value < 1:
        return False

    # Допуск +-10% к элементам ряда
    for std in STANDARD_CURRENTS_LIST:
        if abs(value - std) <= std * 0.1:
            return True

    return False


def fix_merged_currents(text: str) -> str:
    """
    Разбирает слипшиеся числовые токены и усекает завышенные значения.

    Логика:
        1. Если число > MAX_ALLOWED_CURRENT — пробует отрезать 1 или
           2 цифры с конца, чтобы получить валидный номинал
           (например, 10000А -> 1000А).
        2. Если число валидное и < 5 цифр — возвращает "<val>А".
        3. Иначе — перебирает все варианты разбиения строки на две
           части, ищет пару типовых номиналов:
             - если num1 == num2 — возвращает "<val>А";
             - если пара (min, max) есть в EXACT_RANGES_MAP —
               возвращает соответствующий формализованный диапазон;
             - иначе запоминает best_candidate "<min>-<max>А".
        4. Если ничего не найдено — возвращает исходный text.

    Args:
        text (str): Токен с числом (возможно, слипшимся).

    Returns:
        str: Нормализованное значение ("1000А") или диапазон
            ("160-250А"), либо исходный text.
    """
    if not text:
        return ""

    clean_digits = re.sub(r'\D', '', text)
    if not clean_digits:
        return text

    # --- 1. Проверка одиночного числа с возможным усечением лишних цифр с конца ---
    val_full = int(clean_digits)

    # Если значение слишком велика (> 6300A), то производится
    # попытка отрезать 1 или 2 цифры с конца
    if val_full > MAX_ALLOWED_CURRENT:
        # Попытка обрезать 1 цифру с конца (10000 -> 1000)
        if len(clean_digits) > 1:
            truncated_1 = int(clean_digits[:-1])
            if truncated_1 <= MAX_ALLOWED_CURRENT and is_typical_current(truncated_1):
                return f"{truncated_1}А"

        # Попытка обрезать 2 цифры с конца (100000 -> 1000)
        if len(clean_digits) > 2:
            truncated_2 = int(clean_digits[:-2])
            if truncated_2 <= MAX_ALLOWED_CURRENT and is_typical_current(truncated_2):
                return f"{truncated_2}А"

    # Если исходное число было нормальным
    elif len(clean_digits) < 5 and is_typical_current(val_full):
        return f"{val_full}А"

    # --- 2. Логика разделения слипшихся диапазонов (250160 -> 160-250) ---
    length = len(clean_digits)
    best_candidate = None

    for split_idx in range(1, length):
        part1, part2 = clean_digits[:split_idx], clean_digits[split_idx:]
        part2_clean = part2.lstrip('0')

        if not part2_clean:
            continue

        num1, num2 = int(part1), int(part2_clean)

        if num1 > MAX_ALLOWED_CURRENT or num2 > MAX_ALLOWED_CURRENT:
            continue

        if is_typical_current(num1) and is_typical_current(num2):
            if num1 == num2:
                return f"{num1}А"

            smaller, larger = min(num1, num2), max(num1, num2)
            if (smaller, larger) in EXACT_RANGES_MAP:
                return EXACT_RANGES_MAP[(smaller, larger)]

            best_candidate = f"{smaller}-{larger}А"

    return best_candidate if best_candidate else text


def normalize_current_range(current_range: str) -> str:
    """
    Нормализует диапазон токов до ближайшего стандартного.

    Логика:
        1. Разбивает строку по '-', '–', '—' на две части.
        2. Извлекает числа (поддерживает ',' и '.').
        3. Если пара (min, max) точно есть в EXACT_RANGES_MAP —
           возвращает формализованный текст.
        4. Иначе — ищет ближайший диапазон по скорингу:
             score = min_rel_diff*0.6 + max_rel_diff*0.4 + ratio_diff
           где ratio_diff = |our_ratio - std_ratio| * 50.
           Если best_score < 50 — возвращает best_match.
        5. Иначе — приводит каждый край к ближайшему стандартному
           току; если отклонение < 20% — возвращает "<s>-<l>А"
           (или формализованный диапазон, если пара есть в индексе).
        6. Если ничего не подошло — возвращает исходный current_range.

    Args:
        current_range (str): Диапазон токов (например, "250-630А").

    Returns:
        str: Нормализованный диапазон или исходное значение.
    """
    if not current_range or '-' not in current_range:
        return current_range

    parts = re.split(r'[-–—]', current_range)
    if len(parts) != 2:
        return current_range

    p1_digits = re.sub(r'[^\d.,]', '', parts[0].replace(',', '.'))
    p2_digits = re.sub(r'[^\d.,]', '', parts[1].replace(',', '.'))

    if not (p1_digits and p2_digits):
        return current_range

    try:
        num1, num2 = float(p1_digits), float(p2_digits)
    except ValueError:
        return current_range

    smaller, larger = min(num1, num2), max(num1, num2)

    # 1. Быстрый поиск точного совпадения O(1)
    if (smaller, larger) in EXACT_RANGES_MAP:
        return EXACT_RANGES_MAP[(smaller, larger)]

    # 2. Поиск ближайшего диапазона по скорингу
    best_match = None
    best_score = float('inf')

    for std_min, std_max, std_str in STANDARD_RANGES_DB:
        min_rel_diff = (abs(smaller - std_min) / max(smaller, 1)) * 100
        max_rel_diff = (abs(larger - std_max) / max(larger, 1)) * 100

        our_ratio = larger / max(smaller, 1)
        std_ratio = std_max / max(std_min, 1)
        ratio_diff = abs(our_ratio - std_ratio) * 50

        total_score = (min_rel_diff * 0.6) + (max_rel_diff * 0.4) + ratio_diff

        if total_score < best_score:
            best_score = total_score
            best_match = std_str

    if best_match and best_score < 50:
        return best_match

    # 3. Приведение каждого края к стандартным числам
    def closest_std(val: float) -> float:
        return min(STANDARD_CURRENTS_LIST, key=lambda x: abs(x - val))

    std_s = closest_std(smaller)
    std_l = closest_std(larger)

    if (abs(std_s - smaller) / max(smaller, 1) < 0.20) and (abs(std_l - larger) / max(larger, 1) < 0.20):
        s_res, l_res = min(std_s, std_l), max(std_s, std_l)
        if (s_res, l_res) in EXACT_RANGES_MAP:
            return EXACT_RANGES_MAP[(s_res, l_res)]

        # Преобразуем целые числа без лишних нулей (.0)
        s_str = f"{int(s_res)}" if s_res.is_integer() else f"{s_res}"
        l_str = f"{int(l_res)}" if l_res.is_integer() else f"{l_res}"
        return f"{s_str}-{l_str}А"

    return current_range


def is_leakage_current(token: str) -> bool:
    """
    Проверяет, является ли токен током утечки дифф. автомата/УЗО
    только по синтаксическим правилам (без проверки списка номиналов).

    Критерии:
        - содержит 'MA' или 'МА' (миллиамперы);
        - после снятия префикса характеристики (B/C/D) начинается с '0'
          (например, '003A', '0.03A', '0,03').

    Args:
        token (str): Проверяемый токен.

    Returns:
        bool: True, если токен похож на ток утечки, иначе False.
    """
    if not token:
        return False

    token_upper = token.upper().strip()

    # 1. Явное указание миллиампер (mA / мА)
    if 'MA' in token_upper or 'МА' in token_upper:
        return True

    # 2. Очищаем от букв характеристики (B, C, D), оставив чистую запись
    clean_num_str = re.sub(r'^[BВCСDД]', '', token_upper).strip()

    # Значения, начинающиеся с '0' (например, '003A', '0.03A', '0,03') — токи утечки
    if clean_num_str.startswith('0'):
        return True

    return False


def extract_first_valid_current(text: str) -> str:
    """
    Извлекает рабочий ток по приоритетам синтаксических правил.

    Приоритеты:
        0. Диапазоны с явной 'А' на конце (40-160А).
        1. Ток с явной 'А' на конце (H25А, C16A, 16А, 63A).
        2. Характеристика/префикс + целое без 'А' (B10, C16, H16,
           Ih=16, Iн 100).
        3. Десятичные номиналы (2.5А).
        4. Отдельно стоящие целые числа (16, 20).
        5. Слипшиеся числа (например, "250160").

    На каждом шаге токены, похожие на ток утечки
    (is_leakage_current), пропускаются.

    Args:
        text (str): Сырой текст от OCR.

    Returns:
        str: Извлеченный ток ("16А", "250-630А") или "" если ничего
            не найдено.
    """
    if not text:
        return ""

    # -------------------------------------------------------------------------
    # ПРИОРИТЕТ 0: Диапазоны токов с 'А' на конце (например, 40-160А)
    # -------------------------------------------------------------------------

    p0_range_pattern = r'(?<![A-Za-zА-Яа-я0-9\-])\b(\d+)\s*[\-–—]\s*(\d+)\s*[АA]\b'
    for m in re.finditer(p0_range_pattern, text, re.IGNORECASE):
        token = m.group(0)
        if is_leakage_current(token):
            continue
        raw_range = f"{m.group(1)}-{m.group(2)}А"
        return expand_current_value(raw_range)

    # -------------------------------------------------------------------------
    # ПРИОРИТЕТ 1: Ток с явной 'А' на конце (H16A, C16A, 16А, 63A)
    # -------------------------------------------------------------------------
    # Ищем любую букву/префикс перед числом с 'А', но извлекаем ТОЛЬКО ЧИСЛО m.group(1)

    p1_pattern = r'(?:In|Ih|lH|Iн|I\s*=|I[hн]\s*=|[BВCСDДHН])?\s*(\d{1,4})\s*[АA]\b'
    for m in re.finditer(p1_pattern, text, re.IGNORECASE):
        token = m.group(0)
        if is_leakage_current(token):
            continue
        val = int(m.group(1))
        if val <= MAX_ALLOWED_CURRENT and is_typical_current(val):
            return f"{val}А"

    # -------------------------------------------------------------------------
    # ПРИОРИТЕТ 2: Характеристика/Номинал + целое число БЕЗ 'А' (B10, C16, H16, Ih=16, Iн 100)
    # -------------------------------------------------------------------------
    # Вся префиксная часть (B, C, D, H, Ih=, Iн) объединена в единую незахватываемую группу (?:...)
    # Единственная захватываемая группа (\d{1,4}) — это само число (m.group(1))

    p2_pattern = r'(?:I[hнHН]\s*=?|I\s*=?|[BВCСDДHН])\s*(\d{1,4})\b(?!\s*[\.,\-\/\wАaАa])'

    for m in re.finditer(p2_pattern, text, re.IGNORECASE):
        token = m.group(0)
        if is_leakage_current(token):
            continue

        val_str = m.group(1)
        if not val_str:
            continue

        val = int(val_str)

        if val <= MAX_ALLOWED_CURRENT and is_typical_current(val):
            return f"{val}А"

    # -------------------------------------------------------------------------
    # ПРИОРИТЕТ 3: Десятичные номиналы (1.6А, 2.5А)
    # -------------------------------------------------------------------------

    p3_dec_pattern = r'(?<![A-Za-zА-Яа-я0-9\-])(?:In|Ih|lH|[HН])?\s*(\d+[\.,]\d+)\s*[АA]?(?![A-Za-zА-Яа-я0-9])'
    for m in re.finditer(p3_dec_pattern, text, re.IGNORECASE):
        token = m.group(0)
        if is_leakage_current(token):
            continue
        norm_val_str = m.group(1).replace(',', '.')
        try:
            val_float = float(norm_val_str)
            if val_float <= MAX_ALLOWED_CURRENT and is_typical_current(val_float):
                return expand_current_value(f"{norm_val_str}А")
        except ValueError:
            continue

    # -------------------------------------------------------------------------
    # ПРИОРИТЕТ 4: Отдельно стоящие числа (изолированные пробелами/переносами строк)
    # -------------------------------------------------------------------------

    # 1. Поиск и извлечение: разбиваем весь текст по пробелам и переносам строк
    raw_tokens = [t.strip() for t in re.split(r'[\s\n\r]+', text) if t.strip()]

    # 2. Итерируемся по найденным фрагментам строго сверху вниз
    for token in raw_tokens:
        # Проверка, что фрагмент состоит только из 1-4 цифр
        if not token.isdigit() or not (1 <= len(token) <= 4):
            continue

        # Проверка на ток утечки
        if is_leakage_current(token):
            continue

        val = int(token)

        # Проверка диапазона и вхождение в типичные номиналы
        if val <= MAX_ALLOWED_CURRENT and is_typical_current(val):
            return f"{val}А"

    # -------------------------------------------------------------------------
    # ПРИОРИТЕТ 5: Слипшиеся длинные числа (например, "250160")
    # -------------------------------------------------------------------------
    merged_tokens = re.findall(r'\b\d{4,}[AА]?\b', text)
    for raw_token in merged_tokens:
        fixed = fix_merged_currents(raw_token)
        if fixed and fixed != raw_token:
            return expand_current_diapasons(fixed)

    return ""


def expand_voltage(value):
    """
    Приводит обозначение напряжения к стандартному формату.

    Правила:
        - 'AC\d*/\d*[BВ]' -> 'AC380/415В';
        - '400AC' -> '400 AC'.

    Args:
        value (str): Исходное обозначение напряжения.

    Returns:
        str: Нормализованное обозначение или исходное значение.
    """
    rules = [
        (r'AC\d*/\d*[BВ]$', 'AC380/415', 'В'),
        (r'400AC', '400 ', 'AC')
    ]

    for pattern, multiplier, unit in rules:
        if re.fullmatch(pattern, value):
            return f"{multiplier}{unit}"

    return value


def expand_current_diapasons(token: str) -> str:
    """
    Удаляет OCR-префиксы и нормализует значение/диапазон.

    Логика:
        - Снимает префиксы In/Ih/lH/H/I.
        - Если есть '-' — нормализует диапазон (normalize_current_range).
        - Иначе — извлекает число и возвращает "<val>А" (если <= MAX).

    Args:
        token (str): Токен тока.

    Returns:
        str: Нормализованное значение/диапазон или "" если не удалось.
    """
    if not token:
        return ""

    # Удаление префиксов типа H, lH, In перед цифрами
    cleaned = re.sub(r'^(?:In|Ih|lH|lh|[HНlI])\s*(?=\d|[BВCСDД]\d)', '', token.strip(), flags=re.IGNORECASE)

    if '-' in cleaned:
        return normalize_current_range(cleaned)

    # Выделение числа
    m = re.search(r'\d+(?:[\.,]\d+)?', cleaned)
    if m:
        num_str = m.group(0).replace(',', '.')
        val = float(num_str)
        if val <= MAX_ALLOWED_CURRENT:
            val_str = f"{int(val)}" if val.is_integer() else f"{val}"
            return f"{val_str}А"

    return ""


def expand_current_value(short_value: str) -> str:
    """
    Преобразует сокращенное обозначение тока / исправляет артефакты OCR.

    Логика:
        1. Снимает префиксы In/Ih/lH/H/I.
        2. Если это валидный диапазон "<num>-<num>А" — возвращает
           нормализованный "<num>-<num>А".
        3. Иначе прогоняет через таблицу правил (порядок важен!):
           исправление типичных OCR-ошибок (104 -> 10, 0А -> 10А,
           C1OА -> 10А, 1003А -> 10А, кА-варианты, диапазоны,
           мусорные символы в конце) и общие правила приведения
           к "<num>А".
        4. Если ни одно правило не сработало — возвращает short_value.

    Args:
        short_value (str): Сокращенное/искаженное обозначение тока.

    Returns:
        str: Исправленное значение тока.
    """
    if not short_value:
        return short_value

    short_value = re.sub(r'^(?:In|Ih|lH|lh|[HНlI])\s*(?=\d|[BВCСDД]\d)', '', short_value.strip(), flags=re.IGNORECASE)

    short_value = short_value.strip()

    # 1. Предварительная проверка готовности валидных диапазонов
    if '-' in short_value and (short_value.endswith('А') or short_value.endswith('A')):
        parts = re.split(r'[-–—]', short_value)
        if len(parts) == 2:
            part1 = re.sub(r'[AА]', '', parts[0])
            part2 = re.sub(r'[AА]', '', parts[1])
            if part1.isdigit() and part2.isdigit():
                return f"{part1}-{part2}А"

    # 2. Таблица правил (порядок важен!)
    rules = [
        # --- Новые специфические правила OCR и опечаток ---
        (r'^(\d+)4[AА]?$', lambda m: f"{m.group(1)}А"),        # 104А / 104 -> 10А (4 вместо A)
        (r'^0[AА]$', lambda m: '10А'),           # 0А -> 10А
        (r'^00[AА]$', lambda m: '100А'),         # 00А -> 100А

        # --- Специфические спец-значения и опечатки (высший приоритет) ---
        (r'^0{3}[AА]$', lambda m: '4000А'),
        (r'^04[AА]$', lambda m: '20А'),
        (r'^[0-6][AА]$', lambda m: '6А'),
        (r'^[AА]{1,3}$', lambda m: '6А'),

        # O/Q опечатки в номиналах (C1OА -> C10А -> 10А)
        (r'^[CС]([1-9])[OО][AА]$', lambda m: f"{m.group(1)}0А"),
        (r'^[CС]([1-9]{2})[OО][AА]$', lambda m: f"{m.group(1)}0А"),
        (r'^([1-9])[OО][AА]$', lambda m: f"{m.group(1)}0А"),
        (r'^([1-9]{2})[OО][AА]$', lambda m: f"{m.group(1)}0А"),
        (r'^(\d)0{2}[QO]0[AА]$', lambda m: f"{m.group(1)}000А"),
        (r'^(\d)0[QO][AА]$', lambda m: f"{m.group(1)}0А"),

        # Ошибки OCR вида 1003А -> 10А, 24А -> 2А
        (r'^[0-9]{1,3}0[1-9][AА]$', lambda m: f"{m.group(0)[:2]}А"),
        (r'^\d{2}[4][AА]$', lambda m: f"{m.group(0)[:-2]}А"),
        (r'^\d*[AА]\d[AА]$', lambda m: '6А'),

        # --- Килоамперы (кА) ---
        (r'^0к[AА]$', lambda m: '50кА'),
        (r'^0{2}к[AА]$', lambda m: '100кА'),
        (r'^00к[AА]$', lambda m: '100кА'),
        (r'^5к[AА]$', lambda m: '35кА'),
        (r'^8к[AА]$', lambda m: '85кА'),
        (r'^6к[AА]$', lambda m: '65кА'),
        (r'^2к[AА]$', lambda m: '25кА'),
        (r'^к[AА]$', lambda m: '25кА'),
        (r'^[2-9][1-9][1-9]к[AА]$', lambda m: '100кА'),
        (r'^([1-9])[0OQ]к[AА]$', lambda m: f"{m.group(1)}0кА"),
        (r'^[^1]\d{2}к[AА]$', lambda m: f"{m.group(0)[:-2]}кА"),

        # --- Диапазоны ---
        (r'^0-100[AА]$', lambda m: '40-100А'),
        (r'^0-00[AА]$', lambda m: '160-400А'),
        (r'^[2-4][2-9]0-400[AА]$', lambda m: '320-400А'),
        (r'^[4-6][0-9]0-630[AА]$', lambda m: '504-630А'),
        (r'^[1-9][0-9]{3}-630[AА]$', lambda m: '252-630А'),
        (r'^1[1-9][0-9]-250[AА]$', lambda m: '200-250А'),
        (r'^[4-9][0-9]{2}-800[AА]$', lambda m: '800А'),
        (r'^[3-9]\d{2}-[0-3]\d{2}[AА]$', lambda m: '100-250А'),
        (r'^\b[0]{1,3}-\d{1,4}[AА]\d*\w*$', lambda m: '100-250А'),
        (r'^\d-\d[AА]$', lambda m: '40-100А'),
        (r'^44-100[AА]$', lambda m: '40-100А'),
        (r'^0-125[AА]$', lambda m: '50-125А'),
        (r'^\w\d*-\d*[AА]$', lambda m: f"{m.group(0)[:-1]}А"),

        # --- Мусорные символы на конце ---
        (r'^\d*[AА]{2}$', lambda m: f"{m.group(0)[:-2]}А"),
        (r'^\d*4$', lambda m: f"{m.group(0)[:-1]}А"),

        # --- Общие правила (должны быть В КОНЦЕ списка) ---
        (r'^[BВCСDД]?(\d{1,4})[AА]$', lambda m: f"{m.group(1)}А"),
        (r'^[BВCСDД]?(\d{1,4})$', lambda m: f"{m.group(1)}А"),
    ]

    # 3. Перебор правил
    for pattern, handler in rules:
        match = re.fullmatch(pattern, short_value, flags=re.IGNORECASE)
        if match:
            return handler(match)

    return short_value


def expand_current_name(short_value):
    """
    Приводит сокращенное обозначение имени автомата к полному.

    Правила:
        - 'HG0\\d{2}' -> 'HGD' + последние 2 символа;
        - 'HG0\\d{2}[A-Z]' -> 'HGD' + последние 3 символа.

    Args:
        short_value (str): Сокращенное имя.

    Returns:
        str: Полное имя или исходное значение.
    """
    rules = [
        (r'HG0\d{2}', 'HGD' + short_value[-2:]),
        (r'HG0\d{2}[A-Z]', 'HGD' + short_value[-3:]),
    ]

    for pattern, multiplier in rules:
        if re.fullmatch(pattern, short_value):
            return f"{multiplier}"

    return short_value


def _correct_single_transformer(text):
    """
    Исправляет OCR-ошибки в обозначении одного трансформатора тока.

    Применяет таблицу правил (порядок важен!) для типичных искажений:
        - 'ТТАТ' -> '1ТА1';
        - '31А1' -> '3ТА1';
        - '21А3' -> '2ТА3';
        - '1ТАТ' -> '1ТА1';
        - '11А3' -> '1ТА3';
        - '2Т1' -> '2ТА1';
        - и т.п.

    Если ни одно правило не сработало, пытается извлечь базовую
    структуру через regex: первая цифра [1-4] и вторая [1-6] →
    "<X>ТА<Y>". Если и это не удалось — возвращает исходный text.

    Args:
        text (str): Обозначение одного трансформатора.

    Returns:
        str: Исправленное обозначение ("1ТА1") или исходный text.
    """
    if not text:
        return text

    text = text.strip().upper()

    rules = [
        # ТТАТ → 1ТА1 (первая Т → 1, вторая Т → 1)
        (r'^ТТАТ$', '1ТА1'),
        (r'^ТТА([1-6])$', r'1ТА\1'),
        (r'^Т([1-4])АТ$', r'\1ТА1'),

        # Правила для 31А1 → 3ТА1 (перестановка цифр)
        (r'^31[АA]([1-6])$', r'3ТА\1'),  # Добавлена латинская A
        (r'^13[АA]([1-6])$', r'1ТА\1'),  # Добавлена латинская A

        # Правила для 21А3 → 2ТА3 (лишняя цифра 1) - ИСПРАВЛЕНО!
        (r'^([1-4])1[АA]([1-6])$', r'\1ТА\2'),  # Добавлена латинская A
        (r'^([1-4])2[АA]([1-6])$', r'\1ТА\2'),  # Добавлена латинская A
        (r'^([1-4])3[АA]([1-6])$', r'\1ТА\2'),  # Добавлена латинская A
        (r'^([1-4])\d[АA]([1-6])$', r'\1ТА\2'),  # Общее правило для любой цифры + латинская A

        # Правила для 1ТАТ → 1ТА1 (последняя Т → 1)
        (r'^([1-4])ТАТ$', r'\1ТА1'),
        (r'^([1-4])[ТT]A[TТ]$', r'\1ТА1'),  # Добавлено для латинской A

        # Правила для 11А3 → 1ТА3
        (r'^11[АA]([1-6])$', r'1ТА\1'),  # Добавлена латинская A

        # Общие правила замены
        (r'^ТТ$', '1ТА1'),
        (r'^ТА$', '1ТА1'),
        (r'^[ТT]([1-4])$', r'\1ТА1'),  # Добавлена латинская T
        (r'^([1-4])[ТT]$', r'\1ТА1'),  # Добавлена латинская T

        # Замена А на ТА (исправлено для латинских букв)
        (r'^([1-4])[АA]([1-6])$', r'\1ТА\2'),  # Добавлена латинская A

        # Замена лишних символов (исправлено для латинских букв)
        (r'^([1-4])[ТT][ТT]{1,2}([1-6])$', r'\1ТА\2'),  # Добавлена латинская T

        # Корректные значения оставляем как есть (исправлено для латинских букв)
        (r'^[1-4][ТT][АA][1-6]$', lambda x: x),

        # Правила для смешанных символов (исправлено для латинских букв)
        (r'^([1-4])[ТT][AА][TТ]$', r'\1ТА1'),

        # Новые правила для различных ошибок (исправлено для латинских букв)
        (r'^([1-4])[ТT][АA](\d)$', r'\1ТА\2'),  # 2ТА1 → 2ТА1
        (r'^([1-4])[ТT](\d)$', r'\1ТА\2'),      # 2Т1 → 2ТА1
        (r'^([1-4])[АA](\d)$', r'\1ТА\2'),      # 2А1 → 2ТА1

        # Дополнительное правило для случаев типа 21A3
        (r'^([1-4])(\d)[АA](\d)$', r'\1ТА\3'),  # 21A3 → 2ТА3
    ]

    for pattern, replacement in rules:
        match = re.fullmatch(pattern, text)
        if match:
            if callable(replacement):
                return replacement(text)
            return re.sub(pattern, replacement, text)

    # Если не найдено подходящее правило, попытка извлечь базовую структуру
    base_match = re.match(r'^.*?([1-4]).*?([1-6]).*$', text)
    if base_match:
        return f"{base_match.group(1)}ТА{base_match.group(2)}"

    return text


def search_rcd_or_diff(text):
    """
    Проверяет, относится ли текст к УЗО, диффавтомату или прочим
    коммутационным аппаратам (QS/KM/QSG).

    Логика:
        - Проверяет rcd_patterns (УЗО), diff_patterns (диффавтоматы),
          other_patterns (QS/KM/QSG) через re.fullmatch.
        - Дополнительно вызывает is_definitely_diff(text_strip).
        - Возвращает True, если сработал хотя бы один паттерн.

    Args:
        text (str): Текст для проверки.

    Returns:
        bool: True, если текст похож на УЗО/диф/QS/KM, иначе False.
    """
    text_strip = text.strip()

    # Паттерны для УЗО
    rcd_patterns = [
        r'\b\d+[mM][AА]\b',           # 30mA, 10мА
        r'[CС]?\d+/\d+\.?\d*/\d+',    # 40/0,03/2
        r'\d+\.?\d*\s*[mMмМ]?[AА]',   # 0,03, 0,03А, 0,03мА
        r'УЗО', r'RCD',                # Прямые обозначения
        r'QD\d*', r'QDL\d*',           # Маркировки УЗО
        r'FD\d*', r'FDL\d*',           # Маркировки УЗО
    ]

    # Паттерны для диффавтоматов
    diff_patterns = [
        r'[BCDСВ]\d+[AА]?\s*\d+[mMмМ][AА]',  # C20 30мА, B16 30mA
        r'[BCDСВ]\d+[AА]?/\d+[mMмМ][AА]',    # C20/30мА, B16/30mA
        r'QFD\d*', r'QFDL\d*',                # Обозначения диффавтоматов
        r'ABAT32',                            # Маркировки диффавтоматов
        r'[BCDСВ]\d+[AА]?/\d+\.?\d*/\d+',    # C16/0,03/2
        r'\d+[AА]\s*\d+\.?\d*[mMмМ]?[AА]',   # 16А 0,03А, 20А 30мА
        r'^[0-9]{2}0[0-9][AА]$',
    ]

    other_patterns = [
        r'QS\d+',
        r'KM\d+',
        r'QSG\d+',
    ]

    is_rcd = any(re.fullmatch(pattern, text_strip) for pattern in rcd_patterns)
    is_diff = is_definitely_diff(text_strip)
    is_other = any(re.fullmatch(pattern, text_strip) for pattern in other_patterns)

    return is_rcd or is_diff or is_other


def is_definitely_diff(text: str) -> bool:
    """
    Определяет, относится ли текст к дифференциальному автомату
    или УЗО (RCCB/RCBO).

    Критерии:
        1. Явные обозначения QFD/QFDL/ABAT32/VD/ВД/АД/УЗО/RCBO/RCCB.
        2. Наличие дифференциального тока:
            - миллиамперы (10мА, 30mA);
            - амперы < 1 (0.03А, 0,05A);
            - потерянная точка при OCR (003А, 01А, 03А, 05А);
            - символы IΔn / IDn;
            - дробные серии (40/0,03, 63/0.05, 16/0,1/2).

    Args:
        text (str): Текст для проверки.

    Returns:
        bool: True, если текст похож на диф/УЗО, иначе False.
    """
    if not text:
        return False

    text_upper = text.upper()

    # 1. Явные обозначения типов аппаратов и маркировок УЗО/Диф (QFD, VD, АД, УЗО и т.д.)
    explicit_diff_pattern = r'\b(?:QFD\d*|QFDL\d*|ABAT32|VD\d*|ВД\d*|АД\d*|УЗО|RCBO|RCCB)\b'
    if re.search(explicit_diff_pattern, text_upper):
        return True

    # 2. Поиск явного дифференциального тока (IΔn):
    # - Миллиамперы: 10мА, 30mA, 100мА, 300mA, 500mA
    # - Амперы (дроби): 0.01A, 0,03А, 0.1A, 0.3A
    # - Дробные обозначения серии: 40/0,03 или 16/0.03
    diff_current_patterns = [
        r'\b\d{1,3}\s*[mMмМ][AА]\b',        # Любые миллиамперы: 10mA, 30мА, 100mA, 300мА, 500mA
        r'\b0[\.,]\d{1,3}\s*[AА]\b',       # Любые амперы < 1A: 0.03А, 0,05A, 0.1А, 0,08 A, 0.3A
        r'\b(?:00\d{1,2}|0[135])\s*[AА]\b', # Потерянная точка/запятая при OCR (003А, 01А, 008А, 03А, 05А)
        r'\bI[ΔΔdD]n?\b',                   # Символы IΔn / IDn
        r'\b\d{2,3}/0[\.,]\d{1,3}(?:/\d+)?\b' # Дробные серии: 40/0,03, 63/0.05, 16/0,1/2
    ]

    for pattern in diff_current_patterns:
        if re.search(pattern, text_upper):
            return True

    return False


def is_ta(text):
    """
    Простая проверка на трансформаторы по ключевым паттернам.

    Паттерны:
        - '/5' или '/5A' (вторичный ток 5 А);
        - 'TA1', 'TA2' и т.п.;
        - '300/5', '400/5' и т.п.

    Args:
        text (str): Текст для проверки.

    Returns:
        bool: True, если найден хотя бы один паттерн трансформатора.
    """
    if not text:
        return False

    text_upper = text.upper()

    # Просто проверяем наличие ключевых паттернов трансформаторов
    transformer_indicators = [
        r'/5[АA]?\b',      # содержит /5 или /5A
        r'\bTA\d',         # содержит TA1, TA2 и т.д.
        r'\b\d{3,4}/5',    # содержит 300/5, 400/5
    ]

    for pattern in transformer_indicators:
        if re.search(pattern, text_upper):
            print(f"Найден трансформатор по паттерну '{pattern}': '{text}'")
            return True

    return False


def ta_current_value(text):
    """
    Приводит обозначение одного трансформатора или диапазона
    трансформаторов к корректному виду.

    Логика:
        - Если текст содержит '-', обрабатывает обе части отдельно
          через _correct_single_transformer и возвращает
          "<part1>-<part2>".
        - Иначе — применяет _correct_single_transformer к целому.

    Args:
        text (str): Обозначение трансформатора или диапазона.

    Returns:
        str | None: Исправленное обозначение или None, если text пуст.
    """
    if not text or len(text.strip()) == 0:
        return

    text = text.strip().upper()
    range_pattern = r'^(.+?)\s*[-–—]\s*(.+)$'
    range_match = re.match(range_pattern, text)

    if range_match:
        # Обрабатываем обе части диапазона отдельно
        part1 = range_match.group(1)
        part2 = range_match.group(2)

        corrected_part1 = _correct_single_transformer(part1)
        corrected_part2 = _correct_single_transformer(part2)

        return f"{corrected_part1}-{corrected_part2}"

    return _correct_single_transformer(text)


def search_qf(text: str):
    """
    Ищет и формирует QF-объект (автоматический выключатель) по тексту.

    Логика:
        1. Если текст похож на трансформатор (is_ta) — возвращает
           search_ta(text).
        2. Если текст похож на УЗО/диф/QS/KM (search_rcd_or_diff) —
           возвращает None.
        3. Иначе извлекает регулярками:
            - id_qf (например, QF1, QF2.1);
            - name (BA/BA-, HG..., U...);
            - voltage (AC230, AC400/415, 400AC);
            - current_close (число перед кА);
            - poles (1-4 с P/П/полюс).
        4. Извлекает рабочий ток (extract_first_valid_current).
        5. Если найден хотя бы один атрибут — нормализует диапазон
           (normalize_current_range), обрабатывает voltage/close/name
           через expand_* и создает qf.create_qf.
        6. Возвращает созданный QF или None, если ничего не найдено.

    Args:
        text (str): Сырой текст от OCR.

    Returns:
        qf.QF | None: Созданный объект QF или None.
    """
    if is_ta(text):
        print(f"Найден трансформатор: '{text}'")
        return search_ta(text)

    if search_rcd_or_diff(text):
        print(f"Отсеян УЗО/диффавтомат: '{text}'")
        return None

    # --- 1. Регулярные выражения ---
    id_qf_pattern = re.compile(r"\b\d*[OQ0]F[TG]?\d*\.?\d*\b", re.IGNORECASE)
    name_pattern = re.compile(r"(?:[BВ][АA]\s*\d{1,4}(?:-\d{1,4})?(?:/\d{1,4})?[A-Z]*)(?:-\d{1,2})?|(?:[HН]G[A-Z]?\s*\d{2,4}[A-Z]?)|(?:U[A-Z]{1,3}\d{2,4}[A-Z]?)")
    voltage_pattern = re.compile(r"[AА][CС]\d+/\d{3}|[AА][CС]\s*\d{3}|\b\d{3}[AА][CС]", re.IGNORECASE)
    current_close_pattern = re.compile(r"\d{1,4}(?:[oO0])?\s*(?=[kк][aаAА])", re.IGNORECASE)
    poles_pattern = re.compile(r"\b([1-4])\s*[-_]?\s*(?:[PpПп]|полюс\w*)", re.IGNORECASE)

    # --- 2. Извлечение значений ---
    m_id = id_qf_pattern.search(text)
    device_id = m_id.group(0) if m_id else ''

    m_name = name_pattern.search(text)
    device_name = m_name.group(0).strip() if m_name else ''

    m_poles = poles_pattern.search(text)
    polus = m_poles.group(1) if m_poles else ''

    mount_type = ''

    m_volt = voltage_pattern.search(text)
    current_voltage = m_volt.group(0).strip() if m_volt else ''

    if current_voltage and re.fullmatch(r'[AА][CС]\d+/\d{3}|[AА][CС]\s*\d{3}', current_voltage):
        current_voltage += 'В'

    m_close = current_close_pattern.search(text)

    current_close = (m_close.group(0).strip() + 'кА') if m_close else ''

    # --- 3. Извлечение рабочего тока ---
    current_range = extract_first_valid_current(text)
    print(f"Извлеченный ток: '{current_range}'")

    # --- 4. Проверка на наличие хотя бы одного атрибута ---
    if any([current_range, current_voltage, current_close, device_name]):

        # Нормализация диапазона (если вернулся диапазон)
        if current_range and '-' in current_range:
            print(f"Нормализуем диапазон: '{current_range}'")
            normalized = normalize_current_range(current_range)
            if normalized != current_range:
                current_range = normalized
                print(f"После нормализации: '{current_range}'")

        # Дополнительная обработка / подстановка полного формата
        current_close = expand_current_value(current_close) if current_close else ''
        current_voltage = expand_voltage(current_voltage) if current_voltage else ''
        device_name = expand_current_name(device_name) if device_name else ''

        new_qf = qf.create_qf(
            device_id,
            device_name,
            current_range,
            current_voltage,
            current_close,
            polus,
            mount_type
        )
        new_qf.print_data()

        return new_qf

    return


def search_ta(text):
    """
    Ищет и формирует TA-объект (трансформатор тока) по тексту.

    Логика:
        - Извлекает фрагменты вида "TA1-TA3" или "TA1...TA3"
          через regex ta_text.
        - Если фрагмент найден — нормализует его через
          ta_current_value и создает ta.create_ta.
        - Возвращает созданный TA или None.

    Args:
        text (str): Сырой текст от OCR.

    Returns:
        ta.Trans_TA | None: Созданный объект TA или None.
    """
    ta_text = r"[\dТT]*[ТT]?[АA]+[\dТT]*-[\dТT]*[ТT]?[АA]+[\dТT]*|[\dТT]*[ТT]?[АA]+[\dТT]*...[\dТT]*[ТT]?[АA]+[\dТT]*"
    ta_character = r'^\d{1,2}[0OQ]{1,3}/5[AS]?$'
    ta_exc = ''.join(re.findall(ta_text, text))
    if (ta_exc != ''):
        ta_exc = ta_current_value(ta_exc)
        new_ta = ta.create_ta(ta_exc)
        return new_ta
    return

# if __name__ == "__main__":
#     test_cases = [
#     # Корректные диапазоны
#     "1ТА1-1ТА3",
#     "2ТА4-2ТА6", 
#     "3ТА1-3ТА3",
    
#     # Ошибочные случаи из примеров
#     "31А1-31А3",
#     "1ТАТ-11А3", 
#     "ТТАТ-1ТА3",  # Должно стать: 1ТА1 - 1ТА3
#     "TTA1-1TA3",
#     "2TA1-21A3",
    
#     # Одиночные значения
#     "1ТА1", "2ТА4", "31А1", "1ТАТ", "ТТАТ", "11А3",
    
#     # Дополнительные тесты
#     "ТТА1", "Т1АТ", "1ТТ3", "ТА2", "Т3",
#     "100А", "250-630А", "5кА"
#     ]
#     ta_list = []
#     print("Тестирование функции:")
#     for test in test_cases:
#         corrected = ta_current_value(test)
#         print(f"'{test}' -> '{corrected}'")
#         ta_correct = search_ta(test)
#         if ta_correct != None:
#             ta_list.append(ta_correct)
#     for ta_object in ta_list:
#         print(f"Имя:{ta_object.ta_name}, Количество: {ta_object.ta_quantity}")

