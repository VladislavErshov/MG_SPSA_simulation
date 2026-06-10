"""Simplified drone physics for maneuver-learning task.

No inertia, no wind, constant speed.
On collision: backtrack along saved trajectory, evade by alpha_evade,
then gradually turn back toward target with omega_turn.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

from .obstacles import Obstacle, Circle, Rect, Diamond, Star, Cross, _star_vertices


class Drone:
    """
    2D drone with constant speed and collision-maneuver logic.

    Parameters
    ----------
    start_pos : array-like
    target_pos : array-like
    obstacles : list of Obstacle
    speed : float
        Constant flight speed (m/s).
    dt : float
        Simulation time step (s).
    max_duration : float
        Episode time-out (s).
    target_tolerance : float
        Distance to target considered "reached" (m).
    max_collisions : int
        Maximum collisions before episode abort.
    """

    def __init__(
        self,
        start_pos: Union[list, np.ndarray],
        target_pos: Union[list, np.ndarray],
        obstacles: list[Obstacle],
        speed: float = 5.0,
        dt: float = 0.05,
        max_duration: float = 100.0,
        target_tolerance: float = 1.0,
        max_collisions: int = 20,
    ):
        self.start_pos = np.array(start_pos, dtype=float)
        self.target_pos = np.array(target_pos, dtype=float)
        self.obstacles = obstacles
        self.speed = speed
        self.dt = dt
        self.max_duration = max_duration
        self.target_tolerance = target_tolerance
        self.max_collisions = max_collisions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fly_episode(self, params: dict) -> dict:
        """
        Run one complete flight with given maneuver parameters.

        Parameters
        ----------
        params : dict with keys 'd_back', 'omega_turn', 'alpha_evade'

        Returns
        -------
        dict with keys 'time', 'trajectory', 'n_collisions', 'reached', 'target_pos'
        """
        d_back = float(params["d_back"])
        omega_turn = float(params["omega_turn"])
        alpha_evade = float(params["alpha_evade"])

        pos = self.start_pos.copy()
        direction = self._angle_to_target(pos)

        trajectory = [pos.copy()]
        n_collisions = 0
        time = 0.0

        # Maneuver state machine
        # phase: 0=normal, 1=backtrack, 2=evade, 3=return
        phase = 0
        backtrack_points = []
        backtrack_idx = 0
        evade_start_pos = None
        collision_direction = None

        max_steps = math.ceil(self.max_duration / self.dt)

        for _ in range(max_steps):
            time += self.dt
            prev_pos = pos.copy()

            # --- phase logic ------------------------------------------
            if phase == 0:
                # Normal flight toward target
                direction = self._angle_to_target(pos)
                pos = self._step_forward(pos, direction)

            elif phase == 1:
                # Backtrack along saved points
                if backtrack_idx < len(backtrack_points):
                    target_pt = backtrack_points[backtrack_idx]
                    # Skip points still inside obstacles
                    if self._check_collision(target_pt):
                        backtrack_idx += 1
                        continue
                    direction = np.arctan2(
                        target_pt[1] - pos[1], target_pt[0] - pos[0]
                    )
                    step_vec = self._step_vec(direction)
                    if np.linalg.norm(target_pt - pos) <= np.linalg.norm(step_vec):
                        pos = target_pt.copy()
                        backtrack_idx += 1
                    else:
                        pos = pos + step_vec
                else:
                    # Backtrack finished -> evasion turn
                    phase = 2
                    base_dir = collision_direction if collision_direction is not None else direction
                    direction = base_dir + alpha_evade
                    evade_start_pos = pos.copy()

            elif phase == 2:
                # Fly straight in evasion direction
                pos = self._step_forward(pos, direction)
                # Transition to return after moving a short distance
                if evade_start_pos is not None:
                    dist_evaded = np.linalg.norm(pos - evade_start_pos)
                    if dist_evaded > max(d_back * 0.5, 2.0):
                        phase = 3

            elif phase == 3:
                # Gradually turn back toward target
                target_dir = self._angle_to_target(pos)
                delta = self._normalize_angle(target_dir - direction)
                turn = np.sign(delta) * min(abs(delta), omega_turn * self.dt)
                direction += turn
                pos = self._step_forward(pos, direction)
                # If aligned, resume normal flight
                if abs(delta) < 0.05:
                    phase = 0

            # --- collision detection (segment) ------------------------
            if phase != 1 and self._check_segment_collision(prev_pos, pos):
                n_collisions += 1
                if n_collisions >= self.max_collisions:
                    return {
                        "time": time,
                        "trajectory": np.array(trajectory),
                        "n_collisions": n_collisions,
                        "reached": False,
                        "target_pos": self.target_pos,
                    }
                phase = 1
                collision_direction = direction
                backtrack_points = self._compute_backtrack(trajectory, d_back * (1 + 0.5 * n_collisions))
                backtrack_idx = 0
                evade_start_pos = None

            trajectory.append(pos.copy())

            # --- termination checks -----------------------------------
            if np.linalg.norm(pos - self.target_pos) < self.target_tolerance:
                return {
                    "time": time,
                    "trajectory": np.array(trajectory),
                    "n_collisions": n_collisions,
                    "reached": True,
                    "target_pos": self.target_pos,
                }

        return {
            "time": time,
            "trajectory": np.array(trajectory),
            "n_collisions": n_collisions,
            "reached": False,
            "target_pos": self.target_pos,
        }

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------
    def _check_collision(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1])
        for obs in self.obstacles:
            if isinstance(obs, Circle):
                c = np.array([obs.x, obs.y])
                if np.linalg.norm(pos - c) < obs.radius:
                    return True
            elif isinstance(obs, Rect):
                if abs(x - obs.x) <= obs.width / 2 and abs(y - obs.y) <= obs.height / 2:
                    return True
            elif isinstance(obs, Diamond):
                hw, hh = obs.width / 2, obs.height / 2
                if abs(x - obs.x) / hw + abs(y - obs.y) / hh <= 1:
                    return True
            elif isinstance(obs, Star):
                vertices = _star_vertices(obs.x, obs.y, 5, obs.radius, obs.radius * 0.4)
                if self._point_in_polygon(pos, vertices):
                    return True
            elif isinstance(obs, Cross):
                if abs(x - obs.x) <= obs.arm and abs(y - obs.y) <= obs.thickness / 2:
                    return True
                if abs(x - obs.x) <= obs.thickness / 2 and abs(y - obs.y) <= obs.arm:
                    return True
        return False

    def _check_segment_collision(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if segment [a, b] intersects any obstacle."""
        for obs in self.obstacles:
            if isinstance(obs, Rect):
                if self._segment_vs_rect(a, b, obs):
                    return True
            elif isinstance(obs, Diamond):
                if self._segment_vs_diamond(a, b, obs):
                    return True
            elif isinstance(obs, Star):
                vertices = _star_vertices(obs.x, obs.y, 5, obs.radius, obs.radius * 0.4)
                if self._point_in_polygon(a, vertices) or self._point_in_polygon(b, vertices):
                    return True
                nv = len(vertices)
                for i in range(nv):
                    if self._segments_intersect(a, b, vertices[i], vertices[(i + 1) % nv]):
                        return True
            elif isinstance(obs, Cross):
                if self._segment_vs_cross(a, b, obs):
                    return True
            elif isinstance(obs, Circle):
                if self._segment_vs_circle(a, b, obs):
                    return True
        return False

    # --- per-shape segment collision helpers --------------------------
    def _segment_vs_rect(self, a: np.ndarray, b: np.ndarray, obs: Rect) -> bool:
        half_w, half_h = obs.width / 2, obs.height / 2
        cx, cy = obs.x, obs.y
        if abs(a[0] - cx) <= half_w and abs(a[1] - cy) <= half_h:
            return True
        if abs(b[0] - cx) <= half_w and abs(b[1] - cy) <= half_h:
            return True
        corners = [
            np.array([cx - half_w, cy - half_h]),
            np.array([cx + half_w, cy - half_h]),
            np.array([cx + half_w, cy + half_h]),
            np.array([cx - half_w, cy + half_h]),
        ]
        for i in range(4):
            if self._segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
                return True
        return False

    def _segment_vs_diamond(self, a: np.ndarray, b: np.ndarray, obs: Diamond) -> bool:
        hw, hh = obs.width / 2, obs.height / 2
        cx, cy = obs.x, obs.y
        if abs(a[0] - cx) / hw + abs(a[1] - cy) / hh <= 1:
            return True
        if abs(b[0] - cx) / hw + abs(b[1] - cy) / hh <= 1:
            return True
        corners = [
            np.array([cx, cy + hh]),
            np.array([cx + hw, cy]),
            np.array([cx, cy - hh]),
            np.array([cx - hw, cy]),
        ]
        for i in range(4):
            if self._segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
                return True
        return False

    def _segment_vs_cross(self, a: np.ndarray, b: np.ndarray, obs: Cross) -> bool:
        cx, cy = obs.x, obs.y
        for w, h in [(2 * obs.arm, obs.thickness), (obs.thickness, 2 * obs.arm)]:
            hw, hh = w / 2, h / 2
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
                if self._segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
                    return True
        return False

    def _segment_vs_circle(self, a: np.ndarray, b: np.ndarray, obs: Circle) -> bool:
        c = np.array([obs.x, obs.y])
        r = obs.radius
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-12:
            return np.linalg.norm(a - c) < r
        t = max(0.0, min(1.0, float(np.dot(c - a, ab)) / ab_len_sq))
        closest = a + t * ab
        return np.linalg.norm(closest - c) < r

    @staticmethod
    def _point_in_polygon(pos: np.ndarray, vertices: list[np.ndarray]) -> bool:
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

    @staticmethod
    def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2

    @staticmethod
    def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    def _segments_intersect(
        self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
    ) -> bool:
        o1 = self._orientation(p1, p2, p3)
        o2 = self._orientation(p1, p2, p4)
        o3 = self._orientation(p3, p4, p1)
        o4 = self._orientation(p3, p4, p2)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and self._on_segment(p1, p3, p2):
            return True
        if o2 == 0 and self._on_segment(p1, p4, p2):
            return True
        if o3 == 0 and self._on_segment(p3, p1, p4):
            return True
        if o4 == 0 and self._on_segment(p3, p2, p4):
            return True
        return False

    # ------------------------------------------------------------------
    # Backtrack helpers
    # ------------------------------------------------------------------
    def _compute_backtrack(
        self, trajectory: list[np.ndarray], d_back: float
    ) -> list[np.ndarray]:
        """Return list of points to visit while backtracking d_back metres.
        Points inside obstacles are excluded."""
        traj = np.array(trajectory)
        if len(traj) < 2:
            return [traj[-1].copy()]

        points = []
        total = 0.0
        idx = len(traj) - 1

        while total < d_back and idx > 0:
            seg = np.linalg.norm(traj[idx] - traj[idx - 1])
            total += seg
            idx -= 1
            pt = traj[idx].copy()
            if not self._check_collision(pt):
                points.append(pt)

        if not points:
            points = [traj[max(0, idx)].copy()]

        points.reverse()
        return points

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------
    def _angle_to_target(self, pos: np.ndarray) -> float:
        return np.arctan2(
            self.target_pos[1] - pos[1],
            self.target_pos[0] - pos[0],
        )

    def _step_forward(self, pos: np.ndarray, direction: float) -> np.ndarray:
        return pos + self._step_vec(direction)

    def _step_vec(self, direction: float) -> np.ndarray:
        return self.speed * self.dt * np.array([np.cos(direction), np.sin(direction)])

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
