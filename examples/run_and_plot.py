"""
Запуск N прогонов MixedOptimizer и визуализация всех траекторий
+ верификация скорости сходимости по синтетическому квадратичному тесту.

Примеры:
    python examples/run_and_plot.py                     # 1 прогон
    python examples/run_and_plot.py --runs 10           # 10 прогонов
    python examples/run_and_plot.py --runs 10 --seed 42
    python examples/run_and_plot.py --config configs/simulation/grid.json --runs 10
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from src.drone_simulator.core.drone import Drone, DroneConfig
from src.drone_simulator.config import load_configs
from src.drone_simulator.optimizers import (
    MixedOptimizer,
    MixedOptimizerConfig,
    BlockConfig,
    TargetFollowingSPSA,
)
from src.drone_simulator.visualization.simulation_viz import draw_obstacle


# ------------------------------------------------------------------
# Drone runs
# ------------------------------------------------------------------
def run_one(seed: int, cfg: dict):
    np.random.seed(seed)

    opt_cfg = MixedOptimizerConfig(**cfg['mixed_optimizer'])
    optimizer = TargetFollowingSPSA(opt_cfg)

    drone_cfg = DroneConfig(optimizer_type='spsa', **cfg['physics'])
    drone = Drone(
        np.array(cfg['initial_position']),
        drone_cfg,
        optimizer_config=optimizer,
    )
    drone.set_target(np.array(cfg['target_position']))
    drone.set_obstacles(cfg['obstacles'])
    drone.set_wind(np.array([cfg['wind']['vx'], cfg['wind']['vy']]))

    max_steps = int(cfg['simulation']['duration'] / cfg['physics']['dt'])
    for _ in range(max_steps):
        drone.step()
        if drone.reached_target(cfg['metrics']['target_tolerance']):
            break

    return drone


# ------------------------------------------------------------------
# Synthetic convergence test
# ------------------------------------------------------------------
def synthetic_convergence(n_runs: int = 20, n_steps: int = 500):
    """
    Decoupled 3-parameter quadratic:
        L(theta) = ||theta - theta_star||^2
    Optimum: w*=5.0, phi*=0.5, wind*=2.0.
    Returns arrays: n_values, mean_error, std_error.
    """
    theta_star = np.array([5.0, 0.5, 2.0])

    def loss(theta):
        return float(np.sum((theta - theta_star) ** 2))

    errors_all = []

    for seed in range(n_runs):
        np.random.seed(seed)
        config = MixedOptimizerConfig(
            a=1.0, c=0.2, burn_in=0, num_perturbations=8,
            speed_min=0.0, speed_max=10.0,
            wind_estimate_min=-5.0, wind_estimate_max=5.0,
        )
        blocks = [
            BlockConfig(slice(0, 1), method="exact", q=0),
            BlockConfig(slice(1, 2), method="spsa_off_center", q=1),
            BlockConfig(slice(2, 3), method="exact", q=0),
        ]
        opt = MixedOptimizer(config, loss, blocks=blocks)
        # Start away from optimum so that all blocks have to work
        opt.theta = np.array([1.0, -1.0, 0.0])

        errors = []
        for _ in range(n_steps):
            opt.step()
            err = float(np.sum((opt.theta - theta_star) ** 2))
            errors.append(err)
        errors_all.append(errors)

    errors_arr = np.array(errors_all)  # shape: (n_runs, n_steps)
    n_values = np.arange(1, n_steps + 1)
    mean_err = np.mean(errors_arr, axis=0)
    std_err = np.std(errors_arr, axis=0)

    return n_values, mean_err, std_err


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_results(drones, base_seed, cfg: dict):
    obstacles = cfg['obstacles']
    target = np.array(cfg['target_position'])
    start = np.array(cfg['initial_position'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"MixedOptimizer — Trajectories + Convergence Verification (N={len(drones)})",
        fontsize=14,
    )

    # ---- [0,0] Trajectories ----
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(drones)))

    for i, drone in enumerate(drones):
        traj = drone.get_trajectory()
        final_dist = np.linalg.norm(drone.position - drone.target_position)
        label = f"run {base_seed + i}: dist={final_dist:.2f}m"
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=1.5, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=30, zorder=5)

    ax.scatter(target[0], target[1], color='red', marker='*', s=200, label='Target', zorder=5)
    ax.scatter(start[0], start[1], color='green', marker='s', s=100, label='Start', zorder=5)

    for obs in obstacles:
        draw_obstacle(ax, obs, alpha=0.2)

    # Wind arrow
    wind = np.array([cfg['wind']['vx'], cfg['wind']['vy']])
    if np.linalg.norm(wind) > 1e-6:
        ax.annotate(
            '',
            xy=start + wind * 2,
            xytext=start,
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
        )
        ax.text(
            start[0] + wind[0] * 2.2,
            start[1] + wind[1] * 2.2,
            f"wind {np.linalg.norm(wind):.1f} m/s",
            color='blue',
            fontsize=9,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectories")
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # ---- [0,1] Statistics ----
    ax2 = axes[0, 1]
    ax2.axis('off')

    stats = []
    for i, drone in enumerate(drones):
        final_dist = float(np.linalg.norm(drone.position - drone.target_position))
        reached = final_dist < cfg['metrics']['target_tolerance']
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

    # ---- [1,0] Synthetic convergence (log-log) ----
    ax3 = axes[1, 0]
    n_vals, mean_err, std_err = synthetic_convergence(n_runs=20, n_steps=500)

    ax3.loglog(n_vals, mean_err, color='blue', linewidth=2, label='Mean error over 20 runs')
    ax3.fill_between(n_vals, mean_err - std_err, mean_err + std_err, color='blue', alpha=0.15)

    # Reference slopes
    ref_05 = mean_err[0] * (n_vals[0] ** 0.5) * (n_vals ** (-0.5))
    ref_10 = mean_err[0] * n_vals[0] * (n_vals ** (-1.0))
    ax3.loglog(n_vals, ref_05, 'k--', linewidth=1.5, label='n^{-1/2}  theory q=1 off-center')
    ax3.loglog(n_vals, ref_10, 'k:', linewidth=1.5, label='n^{-1}    exact gradient only')

    ax3.set_xlabel("Iteration n")
    ax3.set_ylabel("E||theta_n - theta*||^2")
    ax3.set_title("Convergence Verification — Synthetic Quadratic")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')

    # ---- [1,1] Drone loss history (mean across runs) ----
    ax4 = axes[1, 1]
    max_len = max(len(d.optimizer.history['loss']) for d in drones)
    loss_matrix = np.full((len(drones), max_len), np.nan)
    for i, drone in enumerate(drones):
        hist = drone.optimizer.history['loss']
        loss_matrix[i, :len(hist)] = hist

    mean_loss = np.nanmean(loss_matrix, axis=0)
    std_loss = np.nanstd(loss_matrix, axis=0)
    t_vals = np.arange(len(mean_loss)) * cfg['physics']['dt']

    ax4.plot(t_vals, mean_loss, color='green', linewidth=2, label='Mean loss')
    ax4.fill_between(t_vals, mean_loss - std_loss, mean_loss + std_loss, color='green', alpha=0.15)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Loss")
    ax4.set_title("Drone Loss History (mean over runs)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = Path("results/mixed_optimizer_runs.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run MixedOptimizer and plot trajectories")
    parser.add_argument("--config", type=str, default="configs/simulation/default.json",
                        help="Path to simulation JSON config (default: configs/simulation/default.json)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (default: 0)")
    args = parser.parse_args()

    cfg = load_configs(args.config)

    drones = []
    for i in range(args.runs):
        drone = run_one(args.seed + i, cfg)
        drones.append(drone)
        print(f"run {args.seed + i}: "
              f"dist={np.linalg.norm(drone.position - drone.target_position):.2f} "
              f"traj={drone.get_trajectory_length():.2f} "
              f"coll={drone.get_collision_count()} "
              f"time={drone.time:.2f}")

    plot_results(drones, args.seed, cfg)


if __name__ == "__main__":
    main()
