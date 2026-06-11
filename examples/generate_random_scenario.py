"""Generate random scenarios for maneuver learning.

Usage:
    # Generate all grid variants
    python examples/generate_random_scenario.py --all-grids --seed 1

    # Generate a single wall scenario
    python examples/generate_random_scenario.py --type wall --out configs/simulation/random_wall.json --seed 42

    # Generate a single grid scenario
    python examples/generate_random_scenario.py --type grid --shape star5 --out configs/simulation/random_stars.json
"""

import argparse
from typing import Literal

from drone_simulator.scenario import generate_grid_scenario, generate_wall_scenario, save_scenario


GRID_SHAPES: list[Literal["circle", "rect", "diamond", "star5", "cross", "crossed_rect", "ellipse", "poly"]] = [
    "circle", "rect", "diamond", "star5", "cross", "crossed_rect", "ellipse", "poly"
]


def generate_all_grids(seed: int) -> None:
    """Generate one scenario for each supported grid shape."""
    for i, shape in enumerate(GRID_SHAPES):
        out_path = f"configs/simulation/grid_{shape}_random.json"
        scenario = generate_grid_scenario(
            shape=shape,
            grid_n=6,
            spacing=5.0,
            obstacle_size=2.0,
            seed=seed + i,
        )
        save_scenario(scenario, out_path)
        print(f"Saved grid_{shape}_random.json  ({len(scenario['obstacles'])} obstacles)")


def main():
    parser = argparse.ArgumentParser(description="Generate random scenarios")
    parser.add_argument(
        "--all-grids", action="store_true",
        help="Generate all grid-shape variants (circle, rect, diamond, star5, cross, crossed_rect)"
    )
    parser.add_argument(
        "--type", type=str, default="wall", choices=["wall", "grid"],
        help="Scenario type"
    )
    parser.add_argument(
        "--shape", type=str, default="circle", choices=GRID_SHAPES,
        help="Grid obstacle shape (only for --type grid)"
    )
    parser.add_argument(
        "--out", type=str, default="configs/simulation/random_scenario.json",
        help="Output path"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed"
    )
    args = parser.parse_args()

    if args.all_grids:
        generate_all_grids(args.seed)
        return

    if args.type == "wall":
        scenario = generate_wall_scenario(seed=args.seed)
    else:
        scenario = generate_grid_scenario(
            shape=args.shape,
            seed=args.seed,
        )
    save_scenario(scenario, args.out)
    print(f"Saved {args.out}  ({len(scenario['obstacles'])} obstacles)")


if __name__ == "__main__":
    main()
