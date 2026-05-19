"""Visualization helpers for drone simulations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def draw_obstacle(ax, obs, color="red", alpha=0.3, fill=True):
    x, y = obs[0], obs[1]
    obs_type = obs[-1] if isinstance(obs[-1], str) else "circle"
    if obs_type == "diamond":
        w, h = obs[2], obs[3]
        hw, hh = w / 2, h / 2
        corners = [(x, y + hh), (x + hw, y), (x, y - hh), (x - hw, y)]
        polygon = plt.Polygon(corners, color=color, alpha=alpha, fill=fill)
        ax.add_patch(polygon)
        ax.plot(x, y, "rx", markersize=8)
    elif obs_type == "star5":
        n = 5
        outer_r = obs[2] * 1.5
        inner_r = obs[2] * 0.6
        angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) - np.pi / 2
        radii = np.tile([outer_r, inner_r], n)
        xs = x + radii * np.cos(angles)
        ys = y + radii * np.sin(angles)
        polygon = plt.Polygon(list(zip(xs, ys)), color=color, alpha=alpha, fill=fill)
        ax.add_patch(polygon)
        ax.plot(x, y, "rx", markersize=8)
    elif obs_type == "star6":
        n = 6
        outer_r = obs[2] * 1.5
        inner_r = obs[2] * 0.75
        angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) - np.pi / 2
        radii = np.tile([outer_r, inner_r], n)
        xs = x + radii * np.cos(angles)
        ys = y + radii * np.sin(angles)
        polygon = plt.Polygon(list(zip(xs, ys)), color=color, alpha=alpha, fill=fill)
        ax.add_patch(polygon)
        ax.plot(x, y, "rx", markersize=8)
    elif obs_type == "rect":
        w, h = obs[2], obs[3]
        rect = Rectangle((x - w / 2, y - h / 2), w, h, color=color, alpha=alpha, fill=fill)
        ax.add_patch(rect)
        ax.plot(x, y, "rx", markersize=8)
    else:
        r = obs[2]
        circle = Circle((x, y), r, color=color, alpha=alpha, fill=fill)
        ax.add_patch(circle)
        ax.plot(x, y, "rx", markersize=8)
