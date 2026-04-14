# tms-dedup

Skill для qwen CLI: поиск семантических дубликатов тест-кейсов в выгрузке TestIT (xlsx).

## Как работает

Гибридный пайплайн из 4 шагов:

```
xlsx → extract → cluster (TF-IDF + MinHash) → verify (LLM) → report.md
```

1. **extract** — парсит xlsx TestIT, нормализует в `tests.json`
2. **cluster** — MinHash-LSH + косинусная близость, собирает группы кандидатов
3. **verify** — батчами отдаёт кластеры LLM, получает severity / master / drop_ids
4. **report** — markdown-отчёт с группами по severity

LLM (qwen-coder-next, 130k ctx) видит только маленькие батчи кластеров — никогда все 2500 тестов сразу.

## Быстрый старт

```bash
pip install -r requirements.txt
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
| `--threshold` | 0.75 | Косинусная близость для TF-IDF |
| `--lsh-threshold` | 0.5 | Порог для MinHash-LSH (первичный фильтр) |
| `--min-size` | 2 | Минимум тестов в группе |
| `--max-size` | 20 | Максимум тестов в группе (крупные режутся) |

## Embedder'ы

Дефолтный — char n-gram TF-IDF (offline, без зависимостей от GPU/API).
Для лучшего recall на перефразах подключи `sbert` или `api` (см. `scripts/embedders/`).
