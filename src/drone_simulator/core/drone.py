"""Simplified drone physics for maneuver-learning task.

No inertia, no wind, constant speed.
On collision: backtrack along saved trajectory, evade by alpha_evade,
then gradually turn back toward target with omega_turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .obstacles import Obstacle


@dataclass(frozen=True)
class ManeuverParams:
    """Typed maneuver parameters."""

    d_back: float
    omega_turn: float
    alpha_evade: float


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
        start_pos: list[float] | np.ndarray,
        target_pos: list[float] | np.ndarray,
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
    def fly_episode(self, params: ManeuverParams) -> dict[str, Any]:
        """
        Run one complete flight with given maneuver parameters.

        Returns
        -------
        dict with keys 'time', 'trajectory', 'n_collisions', 'reached', 'target_pos'
        """
        d_back = float(params.d_back)
        omega_turn = float(params.omega_turn)
        alpha_evade = float(params.alpha_evade)

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
                    if self.check_collision(target_pt):
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
            if phase != 1 and self.check_segment_collision(prev_pos, pos):
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
    def check_collision(self, pos: np.ndarray) -> bool:
        """Check if point collides with any obstacle."""
        return any(obs.contains_point(pos) for obs in self.obstacles)

    def check_segment_collision(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if segment [a, b] intersects any obstacle."""
        return any(obs.segment_intersects(a, b) for obs in self.obstacles)

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
            if not self.check_collision(pt):
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
