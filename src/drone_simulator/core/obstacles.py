"""Obstacle classes with polymorphic collision detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True
    return False


def point_in_polygon(pos: np.ndarray, vertices: list[np.ndarray]) -> bool:
    n = len(vertices)
    inside = False
    x, y = float(pos[0]), float(pos[1])
    j = n - 1
    for i in range(n):
        xi, yi = float(vertices[i][0]), float(vertices[i][1])
        xj, yj = float(vertices[j][0]), float(vertices[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def star_vertices(cx: float, cy: float, n: int, outer_r: float, inner_r: float) -> list[np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) - np.pi / 2
    radii = np.tile([outer_r, inner_r], n)
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return [np.array([x, y]) for x, y in zip(xs, ys)]


def regular_vertices(cx: float, cy: float, n: int, radius: float) -> list[np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return [np.array([x, y]) for x, y in zip(xs, ys)]


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------
class Obstacle(ABC):
    @abstractmethod
    def contains_point(self, pos: np.ndarray) -> bool: ...

    @abstractmethod
    def segment_intersects(self, a: np.ndarray, b: np.ndarray) -> bool: ...


# ------------------------------------------------------------------
# Concrete obstacles
# ------------------------------------------------------------------
@dataclass(frozen=True)
class Circle(Obstacle):
    x: float
    y: float
    radius: float
    kind: Literal["circle"] = "circle"

    def contains_point(self, pos: np.ndarray) -> bool:
        c = np.array([self.x, self.y])
        return np.linalg.norm(pos - c) < self.radius

    def segment_intersects(self, a: np.ndarray, b: np.ndarray) -> bool:
        c = np.array([self.x, self.y])
        r = self.radius
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-12:
            return np.linalg.norm(a - c) < r
        t = max(0.0, min(1.0, float(np.dot(c - a, ab)) / ab_len_sq))
        closest = a + t * ab
        return np.linalg.norm(closest - c) < r


@dataclass(frozen=True)
class Rect(Obstacle):
    x: float
    y: float
    width: float
    height: float
    kind: Literal["rect"] = "rect"

    def _half_sizes(self) -> tuple[float, float]:
        return self.width / 2, self.height / 2

    def contains_point(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1])
        return abs(x - self.x) <= self.width / 2 and abs(y - self.y) <= self.height / 2

    def _corners(self) -> list[np.ndarray]:
        hw, hh = self._half_sizes()
        cx, cy = self.x, self.y
        return [
            np.array([cx - hw, cy - hh]),
            np.array([cx + hw, cy - hh]),
            np.array([cx + hw, cy + hh]),
            np.array([cx - hw, cy + hh]),
        ]

    def segment_intersects(self, a: np.ndarray, b: np.ndarray) -> bool:
        if self.contains_point(a) or self.contains_point(b):
            return True
        corners = self._corners()
        for i in range(4):
            if segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
                return True
        return False


@dataclass(frozen=True)
class Diamond(Obstacle):
    x: float
    y: float
    width: float
    height: float
    kind: Literal["diamond"] = "diamond"

    def contains_point(self, pos: np.ndarray) -> bool:
        hw, hh = self.width / 2, self.height / 2
        x, y = float(pos[0]), float(pos[1])
        return abs(x - self.x) / hw + abs(y - self.y) / hh <= 1

    def _corners(self) -> list[np.ndarray]:
        hw, hh = self.width / 2, self.height / 2
        cx, cy = self.x, self.y
        return [
            np.array([cx, cy + hh]),
            np.array([cx + hw, cy]),
            np.array([cx, cy - hh]),
            np.array([cx - hw, cy]),
        ]

    def segment_intersects(self, a: np.ndarray, b: np.ndarray) -> bool:
        if self.contains_point(a) or self.contains_point(b):
            return True
        corners = self._corners()
        for i in range(4):
            if segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
                return True
        return False


@dataclass(frozen=True)
class Star(Obstacle):
    x: float
    y: float
    radius: float
    kind: Literal["star5"] = "star5"

    def get_vertices(self) -> list[np.ndarray]:
        return star_vertices(self.x, self.y, 5, self.radius, self.radius * 0.4)

    def contains_point(self, pos: np.ndarray) -> bool:
        return point_in_polygon(pos, self.get_vertices())

    def segment_intersects(self, a: np.ndarray, b: np.ndarray) -> bool:
        vertices = self.get_vertices()
        if point_in_polygon(a, vertices) or point_in_polygon(b, vertices):
            return True
        nv = len(vertices)
        for i in range(nv):
            if segments_intersect(a, b, vertices[i], vertices[(i + 1) % nv]):
                return True
        return False


@dataclass(frozen=True)
class Cross(Obstacle):
    x: float
    y: float
    arm: float
    thickness: float
    kind: Literal["cross"] = "cross"

    def _rects(self) -> list[tuple[float, float, float, float]]:
        """Return two (cx, cy, half_w, half_h) for the cross arms."""
        cx, cy = self.x, self.y
        return [
            (cx, cy, self.arm, self.thickness / 2),
            (cx, cy, self.thickness / 2, self.arm),
        ]

    def contains_point(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1])
        for hw, hh in [(self.arm, self.thickness / 2), (self.thickness / 2, self.arm)]:
            if abs(x - self.x) <= hw and abs(y - self.y) <= hh:
                return True
        return False

    def segment_intersects(self, a: np.ndarray, b: np.ndarray) -> bool:
        for cx, cy, hw, hh in self._rects():
            if abs(a[0] - cx) <= hw and abs(a[1] - cy) <= hh:
                return True
            if abs(b[0] - cx) <= hw and abs(b[1] - cy) <= hh:
                return True
            corners = [
                np.array([cx - hw, cy - hh]),
                np.array([cx + hw, cy - hh]),
                np.array([cx + hw, cy + hh]),
                np.array([cx - hw, cy + hh]),
            ]
            for i in range(4):
                if segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
                    return True
        return False


@dataclass(frozen=True)
class Ellipse(Obstacle):
    x: float
    y: float
    rx: float
    ry: float
    kind: Literal["ellipse"] = "ellipse"

    def contains_point(self, pos: np.ndarray) -> bool:
        dx = float(pos[0]) - self.x
        dy = float(pos[1]) - self.y
        return (dx / self.rx) ** 2 + (dy / self.ry) ** 2 < 1

    def segment_intersects(self, a: np.ndarray, b: np.ndarray) -> bool:
        # Fast reject via bounding box
        min_x, max_x = min(a[0], b[0]), max(a[0], b[0])
        min_y, max_y = min(a[1], b[1]), max(a[1], b[1])
        if max_x < self.x - self.rx or min_x > self.x + self.rx:
            return False
        if max_y < self.y - self.ry or min_y > self.y + self.ry:
            return False
        # Check endpoints
        for p in (a, b):
            dx = float(p[0]) - self.x
            dy = float(p[1]) - self.y
            if (dx / self.rx) ** 2 + (dy / self.ry) ** 2 < 1:
                return True
        # Subdivide and sample midpoints (2 levels)
        mids = [(a + b) / 2]
        for _ in range(2):
            next_mids = []
            for m in mids:
                dx = float(m[0]) - self.x
                dy = float(m[1]) - self.y
                if (dx / self.rx) ** 2 + (dy / self.ry) ** 2 < 1:
                    return True
                next_mids.append((a + m) / 2)
                next_mids.append((m + b) / 2)
            mids = next_mids
        return False


@dataclass(frozen=True)
class Poly(Obstacle):
    x: float
    y: float
    radius: float
    n: int
    kind: Literal["poly"] = "poly"

    def get_vertices(self) -> list[np.ndarray]:
        return regular_vertices(self.x, self.y, self.n, self.radius)

    def contains_point(self, pos: np.ndarray) -> bool:
        return point_in_polygon(pos, self.get_vertices())

    def segment_intersects(self, a: np.ndarray, b: np.ndarray) -> bool:
        vertices = self.get_vertices()
        if point_in_polygon(a, vertices) or point_in_polygon(b, vertices):
            return True
        nv = len(vertices)
        for i in range(nv):
            if segments_intersect(a, b, vertices[i], vertices[(i + 1) % nv]):
                return True
        return False


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------
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
    if kind == "ellipse":
        if len(data) < 5:
            raise ValueError(f"Ellipse obstacle requires [x, y, rx, ry, 'ellipse'], got {data}")
        return Ellipse(float(data[0]), float(data[1]), float(data[2]), float(data[3]))
    if kind == "poly":
        if len(data) < 5:
            raise ValueError(f"Poly obstacle requires [x, y, radius, n, 'poly'], got {data}")
        return Poly(float(data[0]), float(data[1]), float(data[2]), int(data[3]))

    # default: circle
    if len(data) < 3:
        raise ValueError(f"Circle obstacle requires [x, y, radius], got {data}")
    return Circle(float(data[0]), float(data[1]), float(data[2]))
