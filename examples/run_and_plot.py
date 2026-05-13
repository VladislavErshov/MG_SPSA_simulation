"""
Запуск N прогонов MixedOptimizer и визуализация всех траекторий.

Примеры:
    python examples/run_and_plot.py          # 1 прогон
    python examples/run_and_plot.py --runs 10    # 10 прогонов
    python examples/run_and_plot.py --runs 10 --seed 42
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from src.drone_simulator.core.drone import Drone, DroneConfig
from src.drone_simulator.config import CONFIG
from src.drone_simulator.optimizers import MixedOptimizerConfig, TargetFollowingSPSA


def run_one(seed: int):
    np.random.seed(seed)

    opt_cfg = MixedOptimizerConfig(**CONFIG['mixed_optimizer'])
    optimizer = TargetFollowingSPSA(opt_cfg)

    drone_cfg = DroneConfig(optimizer_type='spsa', **CONFIG['physics'])
    drone = Drone(
        np.array(CONFIG['initial_position']),
        drone_cfg,
        optimizer_config=optimizer,
    )
    drone.set_target(np.array(CONFIG['target_position']))
    drone.set_obstacles(CONFIG['obstacles'])

    max_steps = int(CONFIG['simulation']['duration'] / CONFIG['physics']['dt'])
    for _ in range(max_steps):
        drone.step()
        if drone.reached_target(CONFIG['metrics']['target_tolerance']):
            break

    return drone


def plot_results(drones, base_seed):
    obstacles = CONFIG['obstacles']
    target = np.array(CONFIG['target_position'])
    start = np.array(CONFIG['initial_position'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"MixedOptimizer Trajectories (N={len(drones)})", fontsize=14)

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(drones)))

    for i, drone in enumerate(drones):
        traj = drone.get_trajectory()
        final_dist = np.linalg.norm(drone.position - drone.target_position)
        label = f"run {base_seed + i}: dist={final_dist:.2f}m"
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=1.5, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=30, zorder=5)

    # Target
    ax.scatter(target[0], target[1], color='red', marker='*', s=200, label='Target', zorder=5)
    # Start
    ax.scatter(start[0], start[1], color='green', marker='s', s=100, label='Start', zorder=5)

    # Obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.2)
        ax.add_patch(circle)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectories")
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Stats table
    ax2 = axes[1]
    ax2.axis('off')

    stats = []
    for i, drone in enumerate(drones):
        final_dist = float(np.linalg.norm(drone.position - drone.target_position))
        reached = final_dist < CONFIG['metrics']['target_tolerance']
        stats.append({
            'run': base_seed + i,
            'reached': 'OK' if reached else 'STUCK',
            'dist': final_dist,
            'traj': float(drone.get_trajectory_length()),
            'coll': int(drone.get_collision_count()),
            'time': float(drone.time),
        })

    ok_count = sum(1 for s in stats if s['reached'] == 'OK')
    summary = (
        f"Success rate: {ok_count}/{len(stats)}\n"
        f"Avg trajectory (OK): {np.mean([s['traj'] for s in stats if s['reached'] == 'OK']):.1f}m\n"
        f"Avg collisions (OK): {np.mean([s['coll'] for s in stats if s['reached'] == 'OK']):.1f}\n"
        f"\n"
        f"{'run':>4} {'status':>6} {'dist':>6} {'traj':>6} {'coll':>4} {'time':>5}\n"
        f"{'-' * 35}\n"
    )
    for s in stats:
        summary += f"{s['run']:>4} {s['reached']:>6} {s['dist']:>6.2f} {s['traj']:>6.1f} {s['coll']:>4} {s['time']:>5.1f}\n"

    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_title("Statistics")

    plt.tight_layout()

    out_path = Path("results/mixed_optimizer_runs.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run MixedOptimizer and plot trajectories")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (default: 0)")
    args = parser.parse_args()

    drones = []
    for i in range(args.runs):
        drone = run_one(args.seed + i)
        drones.append(drone)
        print(f"run {args.seed + i}: "
              f"dist={np.linalg.norm(drone.position - drone.target_position):.2f} "
              f"traj={drone.get_trajectory_length():.2f} "
              f"coll={drone.get_collision_count()} "
              f"time={drone.time:.2f}")

    plot_results(drones, args.seed)


if __name__ == "__main__":
    main()
