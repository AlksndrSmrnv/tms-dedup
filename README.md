# tms-dedup

Skill для qwen CLI: поиск семантических дубликатов тест-кейсов в выгрузке TestIT (xlsx).

## Как работает

Двухсигнальный гибридный пайплайн:

```
xlsx → extract → cluster (два сигнала: title, steps) → verify (LLM) → report.md
```

1. **extract** — парсит xlsx TestIT, нормализует в `tests.json`
2. **cluster** — считает `sim_title` и `sim_steps` независимо, присваивает паре одну из меток:
   - `duplicate_full` — близки по обоим сигналам
   - `duplicate_partial` — умеренно близки
   - `same_intent_diff_flow` — одинаковое название, разные шаги
   - `suspicious_copypaste` — **идентичные шаги, disambiguator в названии** («(по счету)» vs «(по карте)») — почти всегда ошибка копипаста, НЕ дубликат
3. **verify** — батчами отдаёт кластеры LLM, получает severity + kind + master + drop_ids
4. **report** — markdown с 4 секциями по label и таблицами метрик пар

LLM (qwen-coder-next, 130k ctx) видит только маленькие батчи кластеров — никогда все 2500 тестов сразу.

## Почему два сигнала

Если склеить title+steps в один текст, шаги (десятки строк) подавляют сигнал названия (одна строка). Тогда тесты вида

```
ППП-Т1213 Проверка поля карта отправителя (по счету)
ППП-Т3215 Проверка поля карта отправителя (по карте)
```

с идентичными шагами ошибочно помечаются как дубликат. В v2 пара проходит hard-override: одинаковый `stem` + разные непустые `disambig` → `sim_title` принудительно занижен до 0.2, пара уходит в категорию `suspicious_copypaste`.

## Быстрый старт

```bash
pip install -r requirements.txt
# Опционально — русская лемматизация для нормализации названий:
# pip install pymorphy3

python scripts/extract.py input.xlsx --out tests.json
python scripts/cluster.py tests.json --out clusters.json
python scripts/verify.py clusters.json tests.json
# → prompts_out/batch_*.txt, скормить LLM, ответы в responses_out/batch_*.json
python scripts/verify.py clusters.json tests.json  # агрегирует
python scripts/report.py clusters_verified.json tests.json --out report.md
```

## Структура xlsx (TestIT)

| Колонка | Содержимое |
|---|---|
| A | ID теста |
| C | Название теста |
| F | Шаг |
| H | Ожидаемый результат |

Строка с заполненными A и C — заголовок нового теста. Следующие строки — его шаги.

## Параметры кластеризатора

| Параметр | Дефолт | Описание |
|---|---|---|
| `--lsh-title-threshold` | 0.5 | Порог MinHash-LSH по названиям |
| `--lsh-steps-threshold` | 0.5 | Порог MinHash-LSH по шагам |
| `--min-size` | 2 | Минимум тестов в группе |
| `--max-size` | 20 | Максимум тестов в группе (крупные режутся) |

Метки расставляются по фиксированной таблице (`LABEL_RULES` в `scripts/cluster.py`), см. комментарий в модуле.

## Embedder'ы

Дефолтный — char n-gram TF-IDF (offline, без зависимостей от GPU/API).
Для лучшего recall на перефразах подключи `sbert` или `api` (см. `scripts/embedders/`).
