"""Visualization helpers for drone simulations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def _draw_star(
    ax: plt.Axes,
    x: float,
    y: float,
    n: int,
    outer_r: float,
    inner_r: float,
    color: str,
    alpha: float,
    fill: bool,
) -> None:
    angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) - np.pi / 2
    radii = np.tile([outer_r, inner_r], n)
    xs = x + radii * np.cos(angles)
    ys = y + radii * np.sin(angles)
    polygon = plt.Polygon(list(zip(xs, ys)), color=color, alpha=alpha, fill=fill)
    ax.add_patch(polygon)


def draw_obstacle(
    ax: plt.Axes,
    obs: list,
    color: str = "red",
    alpha: float = 0.3,
    fill: bool = True,
) -> None:
    x, y = obs[0], obs[1]
    obs_type = obs[-1] if isinstance(obs[-1], str) else "circle"
    if obs_type == "diamond":
        w, h = obs[2], obs[3]
        hw, hh = w / 2, h / 2
        corners = [(x, y + hh), (x + hw, y), (x, y - hh), (x - hw, y)]
        polygon = plt.Polygon(corners, color=color, alpha=alpha, fill=fill)
        ax.add_patch(polygon)
    elif obs_type == "star5":
        _draw_star(ax, x, y, 5, obs[2] * 1.5, obs[2] * 0.6, color, alpha, fill)
    elif obs_type == "cross":
        arm, t = obs[2], obs[3]
        for w, h in [(2 * arm, t), (t, 2 * arm)]:
            rect = Rectangle((x - w / 2, y - h / 2), w, h, color=color, alpha=alpha, fill=fill)
            ax.add_patch(rect)
    elif obs_type == "rect":
        w, h = obs[2], obs[3]
        rect = Rectangle((x - w / 2, y - h / 2), w, h, color=color, alpha=alpha, fill=fill)
        ax.add_patch(rect)
    else:
        r = obs[2]
        circle = Circle((x, y), r, color=color, alpha=alpha, fill=fill)
        ax.add_patch(circle)
    ax.plot(x, y, "rx", markersize=8)
