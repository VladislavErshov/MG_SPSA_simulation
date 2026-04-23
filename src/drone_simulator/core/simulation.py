"""
Simulation engine.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .drone import Drone
from ..visualization.simulation_viz import SimulationVisualizer


@dataclass
class SimulationConfig:
    """Configuration for simulation"""
    duration: float = 30.0
    dt: float = 0.05
    update_interval: int = 1
    plot_interval: int = 5


class DroneSimulator:
    """Simulator for multiple drones with optimizer-based control"""

    def __init__(self, drones: List[Drone], config: Optional[SimulationConfig] = None):
        self.drones = drones
        self.config = config or SimulationConfig()
        self.current_time = 0.0
        self.step_count = 0
        self.visualizer = SimulationVisualizer(drones)

    def step(self) -> bool:
        if self.current_time >= self.config.duration:
            return False
        if self.step_count % self.config.update_interval == 0:
            for drone in self.drones:
                drone.step()
        self.current_time += self.config.dt
        self.step_count += 1
        return True

    def run(self, visualize: bool = True, save_animation: bool = False):
        if visualize:
            self.visualizer.setup_realtime()

        start_time = time.time()
        frames = []

        while self.step():
            if visualize and self.step_count % self.config.plot_interval == 0:
                self.visualizer.update()
            if save_animation and self.step_count % (self.config.plot_interval * 2) == 0:
                frames.append(self.visualizer.capture_frame())

        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.2f} seconds")

        for i, drone in enumerate(self.drones):
            traj = drone.get_trajectory()
            dist = np.linalg.norm(drone.position - drone.target_position)
            print(f"\nDrone {i + 1}:")
            print(f"  Final position: {drone.position}")
            print(f"  Distance to target: {dist:.2f}m")
            print(f"  Reached target: {drone.reached_target()}")
            print(f"  Trajectory length: {np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.2f}m")

        if visualize:
            self.visualizer.show_final()

        if save_animation and frames:
            from PIL import Image
            os.makedirs('results', exist_ok=True)
            frames_pil = [Image.fromarray(frame) for frame in frames]
            frames_pil[0].save('results/drone_simulation.gif',
                             save_all=True, append_images=frames_pil[1:],
                             duration=50, loop=0)
            print("\nAnimation saved as 'results/drone_simulation.gif'")

        return True
