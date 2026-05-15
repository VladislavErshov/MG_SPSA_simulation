"""Real-time and final visualization for drone simulations."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from ..core.drone import Drone


def draw_obstacle(ax, obs, color="red", alpha=0.3, fill=True):
    x, y, r = obs[0], obs[1], obs[2]
    obs_type = obs[3] if len(obs) > 3 else "circle"
    if obs_type == "star6":
        n = 6
        outer_r = r * 1.5
        inner_r = r * 0.75
        angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) - np.pi / 2
        radii = np.tile([outer_r, inner_r], n)
        xs = x + radii * np.cos(angles)
        ys = y + radii * np.sin(angles)
        polygon = plt.Polygon(list(zip(xs, ys)), color=color, alpha=alpha, fill=fill)
        ax.add_patch(polygon)
        ax.plot(x, y, "rx", markersize=8)
    else:
        circle = Circle((x, y), r, color=color, alpha=alpha, fill=fill)
        ax.add_patch(circle)
        ax.plot(x, y, "rx", markersize=8)


class SimulationVisualizer:
    """Handles all matplotlib-based visualization for a drone simulation."""

    def __init__(self, drones: List[Drone]):
        self.drones = drones
        self.fig = None
        self.axes = None

    def setup_realtime(self):
        """Create and return the live-update figure."""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle("Drone Trajectory Optimization", fontsize=16)
        return self.fig, self.axes

    def update(self):
        """Refresh the real-time 2x2 plot grid."""
        if self.axes is None:
            return
        for ax in self.axes.flat:
            ax.clear()
        self._plot_scene(self.axes[0, 0])
        self._plot_speed(self.axes[0, 1])
        self._plot_direction(self.axes[1, 0])
        self._plot_loss(self.axes[1, 1])
        plt.tight_layout()
        plt.pause(0.001)

    def capture_frame(self):
        """Render the current scene and return an RGB array for GIF export."""
        fig_local, ax_local = plt.subplots(figsize=(8, 8))
        self._plot_scene(ax_local, show=False)
        fig_local.canvas.draw()
        rgba_buffer = np.array(fig_local.canvas.buffer_rgba())
        rgb_frame = rgba_buffer[:, :, :3]
        plt.close(fig_local)
        return rgb_frame

    def show_final(self):
        """Disable interactive mode and show the comprehensive final results."""
        plt.ioff()
        self._show_final_results()

    def _plot_scene(self, ax, show=True):
        colors = ["blue", "red", "green", "orange", "purple"]
        for i, drone in enumerate(self.drones):
            traj = drone.get_trajectory()
            color = colors[i % len(colors)]
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=color,
                alpha=0.6,
                linewidth=1,
                label=f"Drone {i + 1} path",
            )
            ax.plot(
                drone.position[0],
                drone.position[1],
                "o",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            ax.plot(
                drone.target_position[0],
                drone.target_position[1],
                "*",
                color=color,
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=1,
            )
        for obs in getattr(self.drones[0], "obstacles", []):
            draw_obstacle(ax, obs)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Drone Trajectories")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        if show:
            plt.draw()

    def _plot_speed(self, ax):
        for i, drone in enumerate(self.drones):
            ax.plot(drone.time_history, drone.speed_history, label=f"Drone {i + 1}", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Drone Speed")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_direction(self, ax):
        for i, drone in enumerate(self.drones):
            direction_deg = np.degrees(drone.direction_history)
            ax.plot(drone.time_history, direction_deg, label=f"Drone {i + 1}", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Direction (degrees)")
        ax.set_title("Drone Direction")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_loss(self, ax):
        for i, drone in enumerate(self.drones):
            optimizer = drone.optimizer
            if len(optimizer.history["loss"]) > 0:
                iterations = range(len(optimizer.history["loss"]))
                ax.plot(iterations, optimizer.history["loss"], label=f"Drone {i + 1}", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Optimizer Loss History")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _show_final_results(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Final Simulation Results", fontsize=16)
        self._plot_scene(axes[0, 0])
        self._plot_speed(axes[0, 1])
        self._plot_direction(axes[0, 2])
        self._plot_loss(axes[1, 0])
        for i, drone in enumerate(self.drones):
            if len(drone.optimizer.history["parameters"]) > 0:
                params = np.array(drone.optimizer.history["parameters"])
                axes[1, 1].plot(range(len(params)), params[:, 0], label=f"Drone {i + 1} speed")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Speed (m/s)")
        axes[1, 1].set_title("Parameter Convergence (Speed)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        for i, drone in enumerate(self.drones):
            traj = drone.get_trajectory()
            distances = np.linalg.norm(traj - drone.target_position, axis=1)
            axes[1, 2].plot(drone.time_history, distances, label=f"Drone {i + 1}")
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].set_ylabel("Distance to Target (m)")
        axes[1, 2].set_title("Approach to Target")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        for ax in [axes[1, 2]]:
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Target threshold")
        plt.tight_layout()
        plt.show()
