"""Generate random wall scenarios for maneuver learning.

Usage:
    python examples/generate_random_scenario.py --out configs/simulation/random_wall_1.json --seed 1
    python examples/generate_random_scenario.py --n-walls 2 --gap-size 2.5 --seed 42
"""

import argparse

from drone_simulator.scenario import generate_wall_scenario, save_scenario


def main():
    parser = argparse.ArgumentParser(description="Generate random wall scenario")
    parser.add_argument(
        "--out", type=str, default="configs/simulation/random_wall.json",
        help="Output path for the generated scenario JSON"
    )
    parser.add_argument(
        "--arena-size", type=float, default=30.0,
        help="Distance from start to target"
    )
    parser.add_argument(
        "--arena-height", type=float, default=20.0,
        help="Height of the arena"
    )
    parser.add_argument(
        "--n-walls", type=int, default=1, choices=[1, 2],
        help="Number of walls (1 or 2)"
    )
    parser.add_argument(
        "--wall-thickness", type=float, default=1.5,
        help="Thickness of each wall segment"
    )
    parser.add_argument(
        "--gap-size", type=float, default=3.0,
        help="Size of the gap in each wall"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed"
    )
    args = parser.parse_args()

    scenario = generate_wall_scenario(
        arena_size=args.arena_size,
        arena_height=args.arena_height,
        n_walls=args.n_walls,
        wall_thickness=args.wall_thickness,
        gap_size=args.gap_size,
        seed=args.seed,
    )
    save_scenario(scenario, args.out)
    print(f"Saved scenario with {len(scenario['obstacles'])} obstacles to {args.out}")


if __name__ == "__main__":
    main()
