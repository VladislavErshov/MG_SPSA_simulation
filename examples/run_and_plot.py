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
# Рандомизация позиций start/target
# ------------------------------------------------------------------
def randomize_positions(
    arena: Drone,
    scenario: dict,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Сдвигает start/target, сохраняя дистанцию, без попадания в препятствия."""
    start_orig = np.array(scenario["start"], dtype=float)
    target_orig = np.array(scenario["target"], dtype=float)
    distance = float(np.linalg.norm(target_orig - start_orig))
    midpoint = (start_orig + target_orig) / 2

    all_x = [obs[0] for obs in scenario["obstacles"]] + [start_orig[0], target_orig[0]]
    all_y = [obs[1] for obs in scenario["obstacles"]] + [start_orig[1], target_orig[1]]
    span_x = max(all_x) - min(all_x)
    span_y = max(all_y) - min(all_y)

    for _ in range(max_attempts):
        angle = rng.uniform(0, 2 * np.pi)
        dx = rng.uniform(-span_x * 0.25, span_x * 0.25)
        dy = rng.uniform(-span_y * 0.25, span_y * 0.25)

        new_mid = midpoint + np.array([dx, dy])
        direction = np.array([np.cos(angle), np.sin(angle)])
        new_start = new_mid - (distance / 2) * direction
        new_target = new_mid + (distance / 2) * direction

        if not arena._check_collision(new_start) and not arena._check_collision(new_target):
            arena.start_pos = new_start
            arena.target_pos = new_target
            return new_start, new_target

    arena.start_pos = start_orig
    arena.target_pos = target_orig
    return start_orig, target_orig


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
    rng = np.random.default_rng(seed)
    arena = create_arena(scenario)
    config = ManeuverOptimizerConfig()
    optimizer = ManeuverOptimizer(config)
    max_dur = scenario.get("max_duration", 100.0)

    losses = []
    trajectories = []
    traj_meta = []  # (start, target) для каждой траектории

    def loss_fn(theta_dict: dict) -> float:
        result = run_episode(theta_dict, arena)
        return compute_loss(result, max_dur)

    for i in range(n_iterations):
        randomize_positions(arena, scenario, rng)

        grad = optimizer.evaluate(mode, loss_fn)
        params = optimizer._to_dict(optimizer.theta)
        result = run_episode(params, arena)
        loss = compute_loss(result, max_dur)
        losses.append(loss)
        trajectories.append(result["trajectory"])
        traj_meta.append((arena.start_pos.copy(), arena.target_pos.copy()))
        print(
            f"Iter {i + 1:3d}: loss={loss:7.2f}  "
            f"d_back={params['d_back']:.2f}  "
            f"omega={params['omega_turn']:.2f}  "
            f"alpha={params['alpha_evade']:.3f}  "
            f"grad=[{grad[0]:7.2f} {grad[1]:7.2f} {grad[2]:7.2f}]"
        )

    return optimizer, losses, trajectories, traj_meta


# ------------------------------------------------------------------
# Визуализация
# ------------------------------------------------------------------
def _fly_and_measure(params: dict, scenario: dict) -> dict:
    arena = create_arena(scenario)
    result = arena.fly_episode(params)
    result["loss"] = compute_loss(result, scenario.get("max_duration", 100.0))
    return result


def plot_results(
    optimizer: ManeuverOptimizer,
    losses: list,
    trajectories: list,
    traj_meta: list,
    mode: str,
    scenario: dict,
    final_result: dict,
    baseline_result: dict,
    config_path: str = "default",
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.25))
    fig.suptitle(f"Обучение манёвра — режим {mode}", fontsize=14, fontweight="bold")

    # --- [0,0] Траектории: обучение тонко, финал жирно ----------------
    ax = axes[0, 0]

    # Обучающие траектории — тонкие полупрозрачные
    cmap = plt.cm.cool
    n_traj = len(trajectories)
    for i, traj in enumerate(trajectories):
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.5, alpha=0.4, zorder=1)

    # Неизменяемая политика (baseline) — пунктир
    ax.plot(
        baseline_result["trajectory"][:, 0],
        baseline_result["trajectory"][:, 1],
        color="grey",
        linewidth=1.5,
        linestyle="--",
        label="Fixed policy",
        zorder=2,
    )

    # Финальный прогон — жирная линия
    ax.plot(
        final_result["trajectory"][:, 0],
        final_result["trajectory"][:, 1],
        color="blue",
        linewidth=3.0,
        label="Learned policy",
        zorder=3,
    )

    # Маркеры только для финального прогона
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
    ax.set_title("Траектории (обучение — тонкие, финал — жирная)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    pts = [scenario["start"], scenario["target"]]
    for obs in scenario["obstacles"]:
        pts.append([obs[0], obs[1]])
    for res in [baseline_result, final_result]:
        for p in res["trajectory"][::max(1, len(res["trajectory"]) // 30)]:
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

    params = optimizer._to_dict(optimizer.theta)

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
        f"Итераций обучения: {len(losses)}\n\n"
        f"Обученные параметры:\n"
        f"  d_back     = {params['d_back']:.2f}\n"
        f"  omega_turn = {params['omega_turn']:.2f}\n"
        f"  alpha_evade= {params['alpha_evade']:.3f}\n\n"
        f"{_fmt(baseline_result, 'Fixed policy')}\n\n"
        f"{_fmt(final_result, 'Learned policy')}"
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
    args = parser.parse_args()

    scenario = load_scenario(args.config)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}, Iterations: {args.iterations}, Seed: {args.seed}")

    optimizer, losses, trajectories, traj_meta = train(
        args.mode, scenario, args.iterations, args.seed
    )

    # Финальные прогоны на оригинальных позициях конфига
    baseline_cfg = ManeuverOptimizerConfig()
    baseline_params = {
        "d_back": baseline_cfg.d_back_init,
        "omega_turn": baseline_cfg.omega_turn_init,
        "alpha_evade": baseline_cfg.alpha_evade_init,
    }
    baseline_result = _fly_and_measure(baseline_params, scenario)
    final_result = _fly_and_measure(optimizer._to_dict(optimizer.theta), scenario)
    print(
        f"\nFixed policy:  loss={baseline_result['loss']:.2f}  "
        f"time={baseline_result['time']:.1f}s  "
        f"collisions={baseline_result['n_collisions']}"
    )
    print(
        f"Learned policy: loss={final_result['loss']:.2f}  "
        f"time={final_result['time']:.1f}s  "
        f"collisions={final_result['n_collisions']}  "
        f"reached={final_result['reached']}"
    )

    plot_results(
        optimizer, losses, trajectories, traj_meta,
        args.mode, scenario, final_result, baseline_result, args.config,
    )


if __name__ == "__main__":
    main()
