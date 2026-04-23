"""
Single entry point: SPSA vs Gradient Descent drone simulation.

Runs both optimizers side-by-side in a single live simulation,
then displays a final comparison dashboard and saves results.
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.drone_simulator.core import Drone, DroneConfig, DroneSimulator, SimulationConfig
from src.drone_simulator.utils import load_simulation_config


def run_simulation():
    print("=" * 70)
    print("SPSA vs Gradient Descent — Live Simulation")
    print("=" * 70)

    # Load configuration from JSON files
    cfg = load_simulation_config()

    start_pos = cfg["initial_position"]
    target_pos = cfg["target_position"]
    obstacles = cfg["obstacles"]
    physics = cfg["physics"]
    sim_config = cfg["simulation"]
    spsa_optimizer = cfg["spsa_optimizer"]
    gd_optimizer = cfg["gd_optimizer"]

    # Create drones with loaded optimizers
    spsa_drone_cfg = DroneConfig(optimizer_type="spsa", **physics)
    gd_drone_cfg = DroneConfig(optimizer_type="gd", **physics)

    drone_spsa = Drone(start_pos.copy(), spsa_drone_cfg, optimizer_config=spsa_optimizer)
    drone_spsa.set_target(target_pos.copy())
    drone_spsa.set_obstacles(obstacles)

    drone_gd = Drone(start_pos.copy(), gd_drone_cfg, optimizer_config=gd_optimizer)
    drone_gd.set_target(target_pos.copy())
    drone_gd.set_obstacles(obstacles)

    simulator = DroneSimulator([drone_spsa, drone_gd], sim_config)

    start = time.time()
    simulator.run(visualize=True, save_animation=True)
    elapsed = time.time() - start

    # Collect results
    drones = {"SPSA": drone_spsa, "GD": drone_gd}
    results = {}
    for name, drone in drones.items():
        traj = drone.get_trajectory()
        results[name] = {
            "final_distance": float(np.linalg.norm(drone.position - drone.target_position)),
            "trajectory_length": float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))),
            "iterations": len(drone.optimizer.history["loss"]),
            "loss_history": [float(x) for x in drone.optimizer.history["loss"]],
            "speed_history": [float(x) for x in drone.speed_history],
            "direction_history": [float(x) for x in drone.direction_history],
            "time": elapsed,
        }

    # Comparison dashboard (shown after the live window is closed)
    _show_comparison_dashboard(drones, results, obstacles, target_pos, start_pos)

    # Save outputs
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("Results saved:")
    print("  - results/drone_simulation.gif")
    print("  - results/comparison_results.json")
    print("=" * 70)


def _show_comparison_dashboard(drones, results, obstacles, target_pos, start_pos):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("SPSA vs Gradient Descent — Final Comparison", fontsize=16)

    colors = {"SPSA": "blue", "GD": "red"}

    # 1. Trajectories
    for name, drone in drones.items():
        traj = drone.get_trajectory()
        axes[0, 0].plot(
            traj[:, 0], traj[:, 1], color=colors[name], label=name, linewidth=2, alpha=0.7
        )
        axes[0, 0].plot(
            drone.target_position[0],
            drone.target_position[1],
            "*",
            color=colors[name],
            markersize=15,
        )
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.2)
        axes[0, 0].add_patch(circle)
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")
    axes[0, 0].set_title("Trajectories")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis("equal")

    # 2. Loss history
    for name, res in results.items():
        loss_hist = res["loss_history"]
        axes[0, 1].plot(range(len(loss_hist)), loss_hist, color=colors[name], label=name, linewidth=2)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Loss Convergence")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale("log")

    # 3. Speed profiles
    for name, drone in drones.items():
        t = np.arange(len(drone.speed_history)) * 0.05
        axes[0, 2].plot(t, drone.speed_history, color=colors[name], label=name, linewidth=2)
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Speed (m/s)")
    axes[0, 2].set_title("Speed Profiles")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Distance to target
    for name, drone in drones.items():
        traj = drone.get_trajectory()
        distances = np.linalg.norm(traj - drone.target_position, axis=1)
        t = np.arange(len(distances)) * 0.05
        axes[1, 0].plot(t, distances, color=colors[name], label=name, linewidth=2)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Distance (m)")
    axes[1, 0].set_title("Approach to Target")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Bar chart comparison
    metrics = ["final_distance", "trajectory_length"]
    spsa_vals = [results["SPSA"][m] for m in metrics]
    gd_vals = [results["GD"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width / 2, spsa_vals, width, label="SPSA", color="blue", alpha=0.7)
    axes[1, 1].bar(x + width / 2, gd_vals, width, label="GD", color="red", alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(["Final Distance", "Trajectory Length"])
    axes[1, 1].set_ylabel("Meters")
    axes[1, 1].set_title("Performance Metrics")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Summary text
    axes[1, 2].axis("off")
    summary = (
        f"SPSA:\n"
        f"  Time: {results['SPSA']['time']:.3f}s\n"
        f"  Final dist: {results['SPSA']['final_distance']:.2f}m\n"
        f"  Trajectory: {results['SPSA']['trajectory_length']:.2f}m\n"
        f"  Iterations: {results['SPSA']['iterations']}\n\n"
        f"Gradient Descent:\n"
        f"  Time: {results['GD']['time']:.3f}s\n"
        f"  Final dist: {results['GD']['final_distance']:.2f}m\n"
        f"  Trajectory: {results['GD']['trajectory_length']:.2f}m\n"
        f"  Iterations: {results['GD']['iterations']}\n\n"
        f"Key differences:\n"
        f"  - SPSA: 2 loss evals/step (stochastic)\n"
        f"  - GD: 4 loss evals/step (deterministic, d=2)\n"
        f"  - GD computes exact numerical gradient\n"
        f"  - SPSA uses simultaneous perturbation"
    )
    axes[1, 2].text(
        0.1,
        0.9,
        summary,
        transform=axes[1, 2].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1, 2].set_title("Summary")

    plt.tight_layout()
    plt.savefig("results/spsa_vs_gd_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_simulation()
