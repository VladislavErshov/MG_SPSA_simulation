"""Visualization helpers for drone simulations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon

from drone_simulator.core.obstacles import Circle as CircleObs, Rect, Diamond, Star, Cross


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
    polygon = Polygon(list(zip(xs, ys)), color=color, alpha=alpha, fill=fill)
    ax.add_patch(polygon)


def draw_obstacle(
    ax: plt.Axes,
    obs: CircleObs | Rect | Diamond | Star | Cross,
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
        _draw_star(ax, x, y, 5, obs.radius * 1.5, obs.radius * 0.6, color, alpha, fill)
    elif isinstance(obs, Cross):
        for w, h in [(2 * obs.arm, obs.thickness), (obs.thickness, 2 * obs.arm)]:
            rect = Rectangle((x - w / 2, y - h / 2), w, h, color=color, alpha=alpha, fill=fill)
            ax.add_patch(rect)
    elif isinstance(obs, Rect):
        w, h = obs.width, obs.height
        rect = Rectangle((x - w / 2, y - h / 2), w, h, color=color, alpha=alpha, fill=fill)
        ax.add_patch(rect)
    else:
        r = obs.radius
        circle = Circle((x, y), r, color=color, alpha=alpha, fill=fill)
        ax.add_patch(circle)
    ax.plot(x, y, "rx", markersize=8)
