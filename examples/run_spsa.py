"""
Запуск одного дрона с MixedOptimizer (SPSA + exact gradient).

Пример:
    python examples/run_spsa.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.drone_simulator.core.drone import Drone, DroneConfig
from src.drone_simulator.config import CONFIG
from src.drone_simulator.optimizers import MixedOptimizerConfig, TargetFollowingSPSA


def run_single(seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    # Конфиг оптимизатора из unified config
    opt_cfg = MixedOptimizerConfig(**CONFIG['mixed_optimizer'])
    optimizer = TargetFollowingSPSA(opt_cfg)

    # Конфиг дрона
    drone_cfg = DroneConfig(
        optimizer_type='spsa',
        **CONFIG['physics'],
    )

    drone = Drone(
        np.array(CONFIG['initial_position']),
        drone_cfg,
        optimizer_config=optimizer,
    )
    drone.set_target(np.array(CONFIG['target_position']))
    drone.set_obstacles(CONFIG['obstacles'])
    drone.set_wind(np.array([CONFIG['wind']['vx'], CONFIG['wind']['vy']]))

    # Симуляция
    max_steps = int(CONFIG['simulation']['duration'] / CONFIG['physics']['dt'])
    for _ in range(max_steps):
        drone.step()
        if drone.reached_target(CONFIG['metrics']['target_tolerance']):
            break

    # Результат
    print(f"seed={seed or 'none'}: "
          f"dist={np.linalg.norm(drone.position - drone.target_position):.2f} "
          f"traj={drone.get_trajectory_length():.2f} "
          f"coll={drone.get_collision_count()} "
          f"time={drone.time:.2f}")

    return drone


if __name__ == "__main__":
    # Один прогон
    drone = run_single()

    # Или Monte Carlo:
    # for s in range(10):
    #     run_single(seed=s)
