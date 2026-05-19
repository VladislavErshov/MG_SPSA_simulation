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
import json
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
    Returns arrays + regression metrics.
    """
    theta_star = np.array([5.0, 0.5, 2.0])

    def loss(theta):
        return float(np.sum((theta - theta_star) ** 2))

    errors_all = []
    dir_errors_all = []

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
        opt.theta = np.array([1.0, -1.0, 0.0])

        errors = []
        dir_errors = []
        for _ in range(n_steps):
            opt.step()
            err = float(np.sum((opt.theta - theta_star) ** 2))
            dir_err = float((opt.theta[1] - theta_star[1]) ** 2)
            errors.append(err)
            dir_errors.append(dir_err)
        errors_all.append(errors)
        dir_errors_all.append(dir_errors)

    errors_arr = np.array(errors_all)
    dir_errors_arr = np.array(dir_errors_all)
    n_values = np.arange(1, n_steps + 1)
    mean_err = np.mean(errors_arr, axis=0)
    std_err = np.std(errors_arr, axis=0)
    mean_err_dir = np.mean(dir_errors_arr, axis=0)
    std_err_dir = np.std(dir_errors_arr, axis=0)

    # Log-log regression on total error and direction-only error
    skip = 50
    log_n = np.log(n_values[skip:])

    log_err_total = np.log(mean_err[skip:])
    slope_total, _ = np.polyfit(log_n, log_err_total, 1)
    pred_total = slope_total * log_n + np.polyfit(log_n, log_err_total, 1)[1]
    ss_res_total = np.sum((log_err_total - pred_total) ** 2)
    ss_tot_total = np.sum((log_err_total - np.mean(log_err_total)) ** 2)
    r2_total = 1.0 - ss_res_total / ss_tot_total if ss_tot_total > 0 else 0.0

    log_err_dir = np.log(mean_err_dir[skip:])
    slope_dir, intercept_dir = np.polyfit(log_n, log_err_dir, 1)
    pred_dir = slope_dir * log_n + intercept_dir
    ss_res_dir = np.sum((log_err_dir - pred_dir) ** 2)
    ss_tot_dir = np.sum((log_err_dir - np.mean(log_err_dir)) ** 2)
    r2_dir = 1.0 - ss_res_dir / ss_tot_dir if ss_tot_dir > 0 else 0.0

    return (n_values, mean_err, std_err, mean_err_dir, std_err_dir,
            slope_total, r2_total, slope_dir, r2_dir)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_results(drones, base_seed, cfg: dict):
    obstacles = cfg['obstacles']
    target = np.array(cfg['target_position'])
    start = np.array(cfg['initial_position'])
    tolerance = cfg['metrics']['target_tolerance']
    dt = cfg['physics']['dt']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.25))
    fig.suptitle(
        f"Mixed-Gradient SPSA — Simulation Results (N={len(drones)} runs)",
        fontsize=14,
        fontweight='bold',
    )

    # ---- Compute stats ----
    stats = []
    for i, drone in enumerate(drones):
        final_dist = float(np.linalg.norm(drone.position - drone.target_position))
        reached = final_dist < tolerance
        stats.append({
            'run': base_seed + i,
            'reached': reached,
            'dist': final_dist,
            'traj': float(drone.get_trajectory_length()),
            'coll': int(drone.get_collision_count()),
            'time': float(drone.time),
        })

    ok_count = sum(1 for s in stats if s['reached'])
    ok_stats = [s for s in stats if s['reached']]

    # ---- [0,0] Representative Trajectories ----
    ax = axes[0, 0]
    # pick best / median / worst by final distance
    sorted_idx = np.argsort([s['dist'] for s in stats])
    rep_indices = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]
    rep_labels = ['Best', 'Median', 'Worst']
    rep_colors = ['#2ca02c', '#ff7f0e', '#d62728']

    for idx, label, color in zip(rep_indices, rep_labels, rep_colors):
        drone = drones[idx]
        traj = drone.get_trajectory()
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5,
                label=label, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker='o', s=30,
                   zorder=4, edgecolors='white', linewidths=0.5)

    ax.scatter(target[0], target[1], color='red', marker='*', s=200,
               label='Target', zorder=4, edgecolors='white', linewidths=0.5)
    ax.scatter(start[0], start[1], color='green', marker='s', s=80,
               label='Start', zorder=4, edgecolors='white', linewidths=0.5)

    for obs in obstacles:
        draw_obstacle(ax, obs, alpha=0.15)

    wind = np.array([cfg['wind']['vx'], cfg['wind']['vy']])
    if np.linalg.norm(wind) > 1e-6:
        ax.text(
            0.02, 0.98,
            f"Wind: {np.linalg.norm(wind):.1f} m/s",
            transform=ax.transAxes,
            color='blue',
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Representative Trajectories")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # ---- [0,1] Performance Metrics ----
    ax2 = axes[0, 1]
    ax2.axis('off')
    ax2.set_title("Performance Metrics", fontsize=12, fontweight='bold', pad=10)

    time_mean = np.mean([s['time'] for s in ok_stats]) if ok_stats else 0.0
    time_std = np.std([s['time'] for s in ok_stats]) if len(ok_stats) > 1 else 0.0
    coll_mean = np.mean([s['coll'] for s in stats])
    coll_std = np.std([s['coll'] for s in stats]) if len(stats) > 1 else 0.0
    traj_mean = np.mean([s['traj'] for s in ok_stats]) if ok_stats else 0.0
    traj_std = np.std([s['traj'] for s in ok_stats]) if len(ok_stats) > 1 else 0.0

    scoreboard = (
        f"Success Rate:        {ok_count / len(stats) * 100:.1f}%\n"
        f"Avg Time to Target:  {time_mean:.1f} ± {time_std:.1f} s\n"
        f"Avg Collisions:      {coll_mean:.1f} ± {coll_std:.1f}\n"
        f"Avg Trajectory:      {traj_mean:.1f} ± {traj_std:.1f} m"
    )
    ax2.text(
        0.5, 0.5, scoreboard,
        transform=ax2.transAxes,
        fontsize=10, fontfamily='monospace',
        verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8),
    )

    # ---- [1,0] Distance to Target vs Time ----
    ax3 = axes[1, 0]
    max_len = max(len(d.get_trajectory()) for d in drones)
    dist_matrix = np.full((len(drones), max_len), np.nan)
    for i, drone in enumerate(drones):
        traj = drone.get_trajectory()
        dists = np.linalg.norm(traj - drone.target_position, axis=1)
        dist_matrix[i, :len(dists)] = dists

    t_vals = np.arange(max_len) * dt
    mean_dist = np.nanmean(dist_matrix, axis=0)
    std_dist = np.nanstd(dist_matrix, axis=0)
    ax3.plot(t_vals, mean_dist, color='darkblue', linewidth=2.5, label='Mean distance')
    ax3.fill_between(t_vals, mean_dist - std_dist, mean_dist + std_dist,
                     color='darkblue', alpha=0.15)
    ax3.axhline(y=tolerance, color='red', linestyle='--', linewidth=1.5,
                label=f'Target tolerance ({tolerance}m)')

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Distance to Target (m)")
    ax3.set_title("Physical Convergence (Distance to Target)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax3.text(0.98, 0.95, f"{ok_count}/{len(stats)} runs reached target",
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ---- [1,1] Synthetic Convergence Verification ----
    ax4 = axes[1, 1]
    (n_vals, mean_err, std_err, mean_err_dir, std_err_dir,
     slope_total, r2_total, slope_dir, r2_dir) = synthetic_convergence(
        n_runs=20, n_steps=500
    )

    ax4.loglog(n_vals, mean_err, color='blue', linewidth=2,
               label='Mean total error')
    ax4.loglog(n_vals, mean_err_dir, color='darkorange', linewidth=2,
               label='Mean direction error (SPSA block)')

    ref_05 = mean_err[0] * (n_vals[0] ** 0.5) * (n_vals ** (-0.5))
    ref_10 = mean_err[0] * n_vals[0] * (n_vals ** (-1.0))
    ax4.loglog(n_vals, ref_05, 'k--', linewidth=1.5,
               label=r'$n^{-1/2}$  theory (q=1 off-center)')
    ax4.loglog(n_vals, ref_10, 'k:', linewidth=1.5,
               label=r'$n^{-1}$    exact gradient only')

    ax4.set_xlabel("Iteration n")
    ax4.set_ylabel(r"$\mathbb{E}\|\theta_n - \theta^*\|^2$")
    ax4.set_title("Convergence Verification — Synthetic Quadratic")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, which='both')

    # Annotation box with regression results
    annot_text = (
        f"Log-log regression (n ≥ 50):\n"
        f"  Total error   slope = {slope_total:.3f}  R² = {r2_total:.3f}\n"
        f"  Direction err slope = {slope_dir:.3f}  R² = {r2_dir:.3f}\n"
        f"  Theory bound (q=1)  ≤ −0.500"
    )
    ax4.text(0.05, 0.35, annot_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    out_path = Path("results/mixed_optimizer_runs.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()

    return slope_total, r2_total, slope_dir, r2_dir, stats


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

    slope_total, r2_total, slope_dir, r2_dir, stats = plot_results(drones, args.seed, cfg)

    ok_count = sum(1 for s in stats if s['reached'])
    ok_stats = [s for s in stats if s['reached']]

    paper_metrics = {
        'synthetic_convergence': {
            'total_error_slope': float(slope_total),
            'total_error_r_squared': float(r2_total),
            'direction_error_slope': float(slope_dir),
            'direction_error_r_squared': float(r2_dir),
            'theory_bound_slope': -0.5,
        },
        'physical_simulation': {
            'n_runs': len(stats),
            'success_rate': ok_count / len(stats) if stats else 0.0,
            'avg_final_distance_m': float(np.mean([s['dist'] for s in stats])) if stats else 0.0,
            'avg_time_to_target_s': float(np.mean([s['time'] for s in ok_stats])) if ok_stats else None,
            'avg_collisions': float(np.mean([s['coll'] for s in stats])) if stats else 0.0,
            'avg_trajectory_length_m': float(np.mean([s['traj'] for s in ok_stats])) if ok_stats else None,
        }
    }

    metrics_path = Path("results/paper_metrics.json")
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(paper_metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    print(f"\nKey numbers for paper:")
    print(f"  Total error   slope = {slope_total:.3f}  R² = {r2_total:.3f}")
    print(f"  Direction err slope = {slope_dir:.3f}  R² = {r2_dir:.3f}  (theory ≤ -0.500)")
    print(f"  Success rate: {ok_count}/{len(stats)} ({paper_metrics['physical_simulation']['success_rate']:.1%})")


if __name__ == "__main__":
    main()
