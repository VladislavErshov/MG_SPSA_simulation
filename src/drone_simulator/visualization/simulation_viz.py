"""Visualization helpers for drone simulations."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse as EllipsePatch, Rectangle, Polygon

from drone_simulator.core.obstacles import Circle as CircleObs, Rect, Diamond, Star, Cross, Ellipse, Poly, parse_obstacle
from drone_simulator.optimizers.spsa import ManeuverOptimizer


def draw_obstacle(
    ax: plt.Axes,
    obs: CircleObs | Rect | Diamond | Star | Cross | Ellipse | Poly,
    color: str = "red",
    alpha: float = 0.3,
    fill: bool = True,
) -> None:
    x, y = obs.x, obs.y
    if isinstance(obs, Diamond):
        hw, hh = obs.width / 2, obs.height / 2
        corners = [(x, y + hh), (x + hw, y), (x, y - hh), (x - hw, y)]
        polygon = Polygon(corners, color=color, alpha=alpha, fill=fill)
        ax.add_patch(polygon)
    elif isinstance(obs, Star):
        verts = obs.get_vertices()
        polygon = Polygon([(v[0], v[1]) for v in verts], color=color, alpha=alpha, fill=fill)
        ax.add_patch(polygon)
    elif isinstance(obs, Cross):
        for w, h in [(2 * obs.arm, obs.thickness), (obs.thickness, 2 * obs.arm)]:
            rect = Rectangle((x - w / 2, y - h / 2), w, h, color=color, alpha=alpha, fill=fill)
            ax.add_patch(rect)
    elif isinstance(obs, Rect):
        w, h = obs.width, obs.height
        rect = Rectangle((x - w / 2, y - h / 2), w, h, color=color, alpha=alpha, fill=fill)
        ax.add_patch(rect)
    elif isinstance(obs, Ellipse):
        patch = EllipsePatch((x, y), 2 * obs.rx, 2 * obs.ry, color=color, alpha=alpha, fill=fill)
        ax.add_patch(patch)
    elif isinstance(obs, Poly):
        verts = obs.get_vertices()
        polygon = Polygon([(v[0], v[1]) for v in verts], color=color, alpha=alpha, fill=fill)
        ax.add_patch(polygon)
    else:
        r = obs.radius
        circle = Circle((x, y), r, color=color, alpha=alpha, fill=fill)
        ax.add_patch(circle)


def plot_results(
    optimizer: ManeuverOptimizer,
    losses: list[float],
    times: list[float],
    trajectories: list[np.ndarray],
    traj_meta: list[tuple[np.ndarray, np.ndarray]],
    mode: str,
    scenario: dict[str, Any],
    final_result: dict[str, Any],
    baseline_result: dict[str, Any],
    config_path: str = "default",
    show_plot: bool = True,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.25))
    fig.suptitle(f"Maneuver learning — mode {mode}", fontsize=14, fontweight="bold")

    # --- [0,0] Trajectories: training thin, final thick ----------------
    ax = axes[0, 0]

    # Training trajectories — thin semi-transparent
    cmap = plt.cm.cool
    n_traj = len(trajectories)
    for i, traj in enumerate(trajectories):
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.5, alpha=0.4, zorder=1)

    # Fixed policy (baseline) — dashed
    ax.plot(
        baseline_result["trajectory"][:, 0],
        baseline_result["trajectory"][:, 1],
        color="grey",
        linewidth=1.5,
        linestyle="--",
        label="Fixed policy",
        zorder=2,
    )

    # Final run — thick line
    ax.plot(
        final_result["trajectory"][:, 0],
        final_result["trajectory"][:, 1],
        color="blue",
        linewidth=3.0,
        label="Learned policy",
        zorder=3,
    )

    # Markers: target (center), fixed start, training starts (semi-transparent)
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

    # Training target positions — small semi-transparent dots
    for i, (_, t_pos) in enumerate(traj_meta):
        color = cmap(i / max(n_traj - 1, 1))
        ax.scatter(
            t_pos[0], t_pos[1], color=color, marker="x", s=20,
            alpha=0.4, zorder=1,
        )

    for obs_data in scenario["obstacles"]:
        draw_obstacle(ax, parse_obstacle(obs_data), alpha=0.15)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectories (training=thin, final=thick)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    # Center view on target with margin based on arena span
    all_pts = [scenario["start"], scenario["target"]]
    for obs_data in scenario["obstacles"]:
        all_pts.append([obs_data[0], obs_data[1]])
    for res in [baseline_result, final_result]:
        for p in res["trajectory"][::max(1, len(res["trajectory"]) // 30)]:
            all_pts.append(p.tolist())
    for s_pos, t_pos in traj_meta:
        all_pts.append(s_pos.tolist())
        all_pts.append(t_pos.tolist())
    all_pts = np.array(all_pts)
    half_span = max(
        all_pts[:, 0].max() - all_pts[:, 0].min(),
        all_pts[:, 1].max() - all_pts[:, 1].min(),
    ) / 2 + 2.0
    mid_x = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
    mid_y = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
    ax.set_xlim(mid_x - half_span, mid_x + half_span)
    ax.set_ylim(mid_y - half_span, mid_y + half_span)

    # --- [0,1] Parameter convergence ------------------------------
    ax2 = axes[0, 1]
    hist = optimizer.history
    thetas = np.array([h["theta"] for h in hist])
    iters = np.arange(1, len(hist) + 1)

    ax2.plot(iters, thetas[:, 0], label="d_back", linewidth=2)
    ax2.plot(iters, thetas[:, 1], label="omega_turn", linewidth=2)
    ax2.plot(iters, thetas[:, 2], label="alpha_evade", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Parameter value")
    ax2.set_title("Parameter convergence")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- [1,0] Loss dynamics -------------------------------------
    ax3 = axes[1, 0]
    iters = range(1, len(losses) + 1)
    ax3.plot(iters, losses, color="darkgreen", linewidth=1.5, alpha=0.6, label="Loss")

    # MA5 smoothed loss
    if len(losses) >= 5:
        ma5 = np.convolve(losses, np.ones(5) / 5, mode="valid")
        ax3.plot(range(3, len(losses) - 1), ma5, color="darkgreen", linewidth=2.5, label="Loss (MA5)")

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Loss", color="darkgreen")
    ax3.tick_params(axis="y", labelcolor="darkgreen")
    ax3.set_title("Loss dynamics")
    ax3.grid(True, alpha=0.3)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(range(1, len(times) + 1), times, color="tab:blue", linewidth=1.5, alpha=0.6, linestyle="--", label="Time (s)")

    # MA5 smoothed time
    if len(times) >= 5:
        ma5_time = np.convolve(times, np.ones(5) / 5, mode="valid")
        ax3_twin.plot(range(3, len(times) - 1), ma5_time, color="tab:blue", linewidth=2.5, linestyle="--", label="Time (MA5)")

    ax3_twin.set_ylabel("Time (s)", color="tab:blue")
    ax3_twin.tick_params(axis="y", labelcolor="tab:blue")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # --- [1,1] Results summary ---------------------------------
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("Final metrics", fontsize=12, fontweight="bold", pad=10)

    params = optimizer.to_dict()

    def _fmt(res: dict, label: str) -> str:
        return (
            f"{label}:\n"
            f"  Loss:        {res['loss']:.2f}\n"
            f"  Time:        {res['time']:.1f} s\n"
            f"  Collisions:  {res['n_collisions']}\n"
            f"  Reached:     {'Yes' if res['reached'] else 'No'}"
        )

    scoreboard = (
        f"Mode: {mode}\n"
        f"Training iterations: {len(losses)}\n\n"
        f"Learned parameters:\n"
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
    out_path = Path(f"results/{cfg_name}.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
