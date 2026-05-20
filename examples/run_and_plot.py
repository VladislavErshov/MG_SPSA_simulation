"""
Эпизодное обучение манёвра дрона с постоянной скоростью.

Usage:
    python examples/run_and_plot.py --mode spsa1 --iterations 30
    python examples/run_and_plot.py --mode spsa2 --iterations 30 --seed 42
    python examples/run_and_plot.py --config configs/simulation/grid.json --mode spsa2 --iterations 30
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.drone_simulator.core.drone import Drone
from src.drone_simulator.optimizers.spsa import ManeuverOptimizer, ManeuverOptimizerConfig
from src.drone_simulator.visualization.simulation_viz import draw_obstacle


# ------------------------------------------------------------------
# Загрузка конфигурации
# ------------------------------------------------------------------
def load_scenario(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Создание арены
# ------------------------------------------------------------------
def create_arena(scenario: dict) -> Drone:
    return Drone(
        start_pos=scenario["start"],
        target_pos=scenario["target"],
        obstacles=scenario["obstacles"],
        speed=scenario.get("speed", 5.0),
        dt=scenario.get("dt", 0.05),
        max_duration=scenario.get("max_duration", 100.0),
    )


# ------------------------------------------------------------------
# Эпизод и лосс
# ------------------------------------------------------------------
def run_episode(params: dict, arena: Drone) -> dict:
    return arena.fly_episode(params)


def compute_loss(result: dict, max_duration: float = 100.0) -> float:
    traj = result["trajectory"]
    traj_len = 0.0
    for i in range(1, len(traj)):
        traj_len += float(np.linalg.norm(traj[i] - traj[i - 1]))
    loss = result["time"] + 0.01 * traj_len + 5.0 * result["n_collisions"]
    if not result["reached"]:
        dist_to_target = float(np.linalg.norm(traj[-1] - result.get("target_pos", traj[-1])))
        loss += 2.0 * dist_to_target
    return loss


# ------------------------------------------------------------------
# Обучение
# ------------------------------------------------------------------
def train(mode: str, scenario: dict, n_iterations: int, seed: int = 0):
    np.random.seed(seed)
    arena = create_arena(scenario)
    config = ManeuverOptimizerConfig()
    optimizer = ManeuverOptimizer(config)
    max_dur = scenario.get("max_duration", 100.0)

    losses = []
    trajectories = []
    def loss_fn(theta_dict: dict) -> float:
        result = run_episode(theta_dict, arena)
        return compute_loss(result, max_dur)

    for i in range(n_iterations):

        grad = optimizer.evaluate(mode, loss_fn)
        params = optimizer.get_params()
        result = run_episode(params, arena)
        loss = compute_loss(result, max_dur)
        losses.append(loss)
        trajectories.append(result["trajectory"])
        print(
            f"Iter {i + 1:3d}: loss={loss:7.2f}  "
            f"d_back={params['d_back']:.2f}  "
            f"omega={params['omega_turn']:.2f}  "
            f"alpha={params['alpha_evade']:.3f}  "
            f"grad=[{grad[0]:7.2f} {grad[1]:7.2f} {grad[2]:7.2f}]"
        )

    return optimizer, losses, trajectories


# ------------------------------------------------------------------
# Визуализация
# ------------------------------------------------------------------
def _fly_and_measure(params: dict, scenario: dict) -> dict:
    arena = create_arena(scenario)
    result = arena.fly_episode(params)
    result["loss"] = compute_loss(result, scenario.get("max_duration", 100.0))
    return result


def plot_results(optimizer: ManeuverOptimizer, losses: list, trajectories: list, mode: str, scenario: dict, config_path: str = "default", show_all: bool = False):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.25))
    fig.suptitle(f"Обучение манёвра — режим {mode}", fontsize=14, fontweight="bold")

    # --- [0,0] Траектории: все итерации ----------------------------
    ax = axes[0, 0]
    baseline_cfg = ManeuverOptimizerConfig()
    baseline_params = {
        "d_back": baseline_cfg.d_back_init,
        "omega_turn": baseline_cfg.omega_turn_init,
        "alpha_evade": baseline_cfg.alpha_evade_init,
    }
    baseline_res = _fly_and_measure(baseline_params, scenario)
    trained_res = _fly_and_measure(optimizer.get_params(), scenario)

    ax.plot(
        baseline_res["trajectory"][:, 0],
        baseline_res["trajectory"][:, 1],
        color="grey",
        linewidth=1.5,
        linestyle="--",
        label="Baseline",
        zorder=2,
    )

    if show_all:
        cmap = plt.cm.cool
        n_traj = len(trajectories)
        for i, traj in enumerate(trajectories):
            color = cmap(i / max(n_traj - 1, 1))
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.5, alpha=0.5, zorder=1)

    ax.plot(
        trained_res["trajectory"][:, 0],
        trained_res["trajectory"][:, 1],
        color="blue",
        linewidth=2.0,
        label="Trained",
        zorder=3,
    )

    target = np.array(scenario["target"])
    start = np.array(scenario["start"])
    ax.scatter(
        target[0], target[1], color="red", marker="*", s=200,
        label="Target", zorder=4, edgecolors="white", linewidths=0.5,
    )
    ax.scatter(
        start[0], start[1], color="green", marker="s", s=80,
        label="Start", zorder=4, edgecolors="white", linewidths=0.5,
    )

    for obs in scenario["obstacles"]:
        draw_obstacle(ax, obs, alpha=0.15)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Траектории" + (" (все итерации)" if show_all else ""))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    # Axis limits from scenario + baseline & trained trajectories
    pts = [scenario["start"], scenario["target"]]
    for obs in scenario["obstacles"]:
        pts.append([obs[0], obs[1]])
    for traj in [baseline_res["trajectory"], trained_res["trajectory"]]:
        for p in traj[::max(1, len(traj) // 50)]:
            pts.append(p.tolist())
    pts = np.array(pts)
    margin = 2.0
    ax.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + margin)
    ax.set_ylim(pts[:, 1].min() - margin, pts[:, 1].max() + margin)

    # --- [0,1] Сходимость параметров ------------------------------
    ax2 = axes[0, 1]
    hist = optimizer.history
    thetas = np.array([h["theta"] for h in hist])
    iters = np.arange(1, len(hist) + 1)

    ax2.plot(iters, thetas[:, 0], label="d_back", linewidth=2)
    ax2.plot(iters, thetas[:, 1], label="omega_turn", linewidth=2)
    ax2.plot(iters, thetas[:, 2], label="alpha_evade", linewidth=2)
    ax2.set_xlabel("Итерация")
    ax2.set_ylabel("Значение параметра")
    ax2.set_title("Сходимость параметров")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- [1,0] Динамика лосса -------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(range(1, len(losses) + 1), losses, color="darkgreen", linewidth=2)
    ax3.set_xlabel("Итерация")
    ax3.set_ylabel("Loss")
    ax3.set_title("Динамика функции потерь")
    ax3.grid(True, alpha=0.3)

    # --- [1,1] Сводка результатов ---------------------------------
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("Итоговые метрики", fontsize=12, fontweight="bold", pad=10)

    def _fmt(res: dict, label: str) -> str:
        return (
            f"{label}:\n"
            f"  Loss:        {res['loss']:.2f}\n"
            f"  Time:        {res['time']:.1f} s\n"
            f"  Collisions:  {res['n_collisions']}\n"
            f"  Reached:     {'Yes' if res['reached'] else 'No'}"
        )

    scoreboard = (
        f"Режим: {mode}\n"
        f"Итераций: {len(losses)}\n\n"
        f"Обученные параметры:\n"
        f"  d_back     = {optimizer.theta[0]:.2f}\n"
        f"  omega_turn = {optimizer.theta[1]:.2f}\n"
        f"  alpha_evade= {optimizer.theta[2]:.3f}\n\n"
        f"{_fmt(baseline_res, 'Baseline')}\n\n"
        f"{_fmt(trained_res, 'Trained')}"
    )
    ax4.text(
        0.5, 0.5, scoreboard,
        transform=ax4.transAxes,
        fontsize=10, fontfamily="monospace",
        verticalalignment="center", horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8),
    )

    plt.tight_layout()
    cfg_name = Path(config_path).stem
    out_path = Path(f"results/maneuver_learning_{cfg_name}.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Episode-based maneuver training")
    parser.add_argument(
        "--config", type=str, default="configs/simulation/default.json",
        help="Path to scenario JSON (default: configs/simulation/default.json)"
    )
    parser.add_argument(
        "--mode", type=str, default="spsa2", choices=["spsa1", "spsa2"],
        help="Режим SPSA: spsa1 (одно измерение) или spsa2 (центрированный)"
    )
    parser.add_argument(
        "--iterations", type=int, default=30,
        help="Число итераций обучения (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)"
    )
    parser.add_argument(
        "--trajectories", type=str, default="best", choices=["best", "all"],
        help="Показ траекторий: best (только лучшая) или all (все итерации)"
    )
    args = parser.parse_args()

    scenario = load_scenario(args.config)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}, Iterations: {args.iterations}, Seed: {args.seed}")
    optimizer, losses, trajectories = train(args.mode, scenario, args.iterations, args.seed)
    plot_results(optimizer, losses, trajectories, args.mode, scenario, args.config, show_all=args.trajectories == "all")


if __name__ == "__main__":
    main()
