# MG SPSA Simulation

## О проекте

Симулятор обучения манёвров БПЛА с использованием гибридного оптимизатора SPSA + exact finite differences.

## Ключевые файлы

- `src/drone_simulator/optimizers/spsa.py` — оптимизатор параметров манёвра
- `src/drone_simulator/core/drone.py` — физика дрона и логика столкновений
- `examples/run_and_plot.py` — обучение и визуализация
- `configs/simulation/` — JSON-сценарии с препятствиями

## Архитектура оптимизатора

`theta = [d_back, omega_turn, alpha_evade]`

- `d_back`, `omega_turn` — exact gradient через central finite difference
- `alpha_evade` — SPSA (spsa1: one-measurement, spsa2: centered two-measurement)

Градиент нормализуется по max absolute component. Step-size decay: `a / (n + A)^p`.

## Важные находки

См. `.claude/memory/spsa_improvements.md` — результаты экспериментов по улучшению сходимости.

Кратко: оригинальный алгоритм лучше модификаций (momentum, gradient clipping, adaptive epsilon не улучшили результат).

## Запуск

```bash
# Один сценарий
python examples/run_and_plot.py --config configs/simulation/grid_star5_random.json --mode spsa2 --iterations 60

# Все сценарии
python .claude/skills/run-all-scenarios/scripts/run_benchmark.py --iterations 60 --workers 4
```

## Зависимости

- Python 3.14+
- numpy, matplotlib

## Скиллы

- `/run-all-scenarios` — бенчмарк всех конфигов
