---
name: run-all-scenarios
description: Запускает обучение на всех сценариях, сравнивает с бейзлайном и перезапускает плохие.
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob]
---

# /run-all-scenarios

Запускает обучение манёвров (SPSA) на всех JSON-сценариях из `configs/simulation/`, сравнивает итоговый loss обученной политики с бейзлайном и при необходимости перезапускает плохие сценарии.

## Использование

```bash
python .claude/skills/run-all-scenarios/scripts/run_benchmark.py [OPTIONS]
```

## Аргументы

- `--iterations` — число итераций обучения (default: 60)
- `--workers` — число параллельных процессов (default: CPU count - 1)
- `--max-retries` — максимальное число перезапусков для одного сценария (default: 3)
- `--runs` — число независимых запусков обучения для усреднения кривых (default: 5)

## Логика работы

1. Собирает все `*.json` из `configs/simulation/`.
2. Запускает `examples/run_and_plot.py --mode spsa2 --iterations 60 --runs 5 --no-display` для каждого конфига в параллельном пуле процессов.
3. Парсит `Fixed policy: loss=...` и `Learned policy: loss=...` из stdout.
4. Если `learned_loss > fixed_loss` (обучение хуже бейзлайна), сценарий отправляется на перезапуск с увеличенным `seed`.
5. Перезапуск повторяется до `--max-retries` раз.
6. Итоговая сводка и JSON с результатами пишутся в `.claude/skills/run-all-scenarios/scripts/benchmark_results.json`.

## Пример

```bash
python .claude/skills/run-all-scenarios/scripts/run_benchmark.py --iterations 60 --workers 4 --max-retries 3 --runs 5
```
