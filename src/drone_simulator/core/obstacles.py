"""Obstacle dataclasses for collision detection and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float
    kind: Literal["circle"] = "circle"


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float
    kind: Literal["rect"] = "rect"


@dataclass(frozen=True)
class Diamond:
    x: float
    y: float
    width: float
    height: float
    kind: Literal["diamond"] = "diamond"


@dataclass(frozen=True)
class Star:
    x: float
    y: float
    radius: float
    kind: Literal["star5"] = "star5"


@dataclass(frozen=True)
class Cross:
    x: float
    y: float
    arm: float
    thickness: float
    kind: Literal["cross"] = "cross"


Obstacle = Circle | Rect | Diamond | Star | Cross


def parse_obstacle(data: list) -> Obstacle:
    """Parse raw obstacle list from JSON config into typed obstacle."""
    if not data:
        raise ValueError("Obstacle data is empty")

    kind = data[-1] if isinstance(data[-1], str) else "circle"

    if kind == "rect":
        if len(data) < 5:
            raise ValueError(f"Rect obstacle requires [x, y, w, h, 'rect'], got {data}")
        return Rect(float(data[0]), float(data[1]), float(data[2]), float(data[3]))
    if kind == "diamond":
        if len(data) < 5:
            raise ValueError(f"Diamond obstacle requires [x, y, w, h, 'diamond'], got {data}")
        return Diamond(float(data[0]), float(data[1]), float(data[2]), float(data[3]))
    if kind == "star5":
        if len(data) < 4:
            raise ValueError(f"Star obstacle requires [x, y, r, 'star5'], got {data}")
        return Star(float(data[0]), float(data[1]), float(data[2]))
    if kind == "cross":
        if len(data) < 5:
            raise ValueError(f"Cross obstacle requires [x, y, arm, t, 'cross'], got {data}")
        return Cross(float(data[0]), float(data[1]), float(data[2]), float(data[3]))

    # default: circle
    if len(data) < 3:
        raise ValueError(f"Circle obstacle requires [x, y, radius], got {data}")
    return Circle(float(data[0]), float(data[1]), float(data[2]))


@staticmethod
def _star_vertices(cx: float, cy: float, n: int, outer_r: float, inner_r: float) -> list[np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) - np.pi / 2
    radii = np.tile([outer_r, inner_r], n)
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return [np.array([x, y]) for x, y in zip(xs, ys)]
