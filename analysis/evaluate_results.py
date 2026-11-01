import csv
import time
from pathlib import Path

import numpy as np
from morphing_udp import udp

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "analysis" / "outputs"

ALGORITHMS = [
    ("Hill Climbing", "hill_climbing_best_rover.npy", 7.417876720428467),
    ("Adaptive Hill Climbing", "adaptive_best_rover.npy", 7.374997138977051),
    ("Genetic Algorithm", "ga_best_rover_new.npy", 6.613618850708008),
    ("Small-Step Morphology Search", "morphology_best_rover_v2.npy", 6.482725620269775),
    ("Parallel Differential Evolution", "multistart_ga_v4_best_rover.npy", 3.743781805038452),
]


def main():
    lower, upper = (np.asarray(bounds, dtype=float) for bounds in udp.get_bounds())
    integer_count = int(udp.get_nix())
    results = []

    print("ESA Morphing Rovers - Five Final Algorithms")
    print("Fitness: lower is better")
    print("Evaluation time: one complete evaluation of 30 ESA scenarios")
    print("=" * 65)

    for name, filename, expected_fitness in ALGORITHMS:
        source = ROOT / "final_results" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing verified rover: {source}")

        rover = np.load(source, allow_pickle=False)
        if rover.shape != lower.shape or not np.all(np.isfinite(rover)):
            raise ValueError(f"Invalid parameter shape or nonfinite values: {source}")
        if np.any(rover < lower) or np.any(rover > upper):
            raise ValueError(f"Parameters outside ESA bounds: {source}")
        if integer_count and not np.array_equal(rover[-integer_count:],
                                                np.trunc(rover[-integer_count:])):
            raise ValueError(f"Noninteger architecture parameters: {source}")

        start = time.perf_counter()
        fitness = float(udp.fitness(rover)[0])
        seconds = time.perf_counter() - start

        if not np.isfinite(fitness) or abs(fitness - expected_fitness) > 1e-5:
            raise ValueError(
                f"Unexpected fitness for {name}: {fitness:.9f} "
                f"(expected approximately {expected_fitness:.9f}). "
                "No comparison CSV was overwritten."
            )

        results.append({
            "Algorithm": name,
            "Fitness": f"{fitness:.12f}",
            "Evaluation Time (s)": f"{seconds:.6f}",
        })
        print(f"{name:<34} Fitness: {fitness:.9f} | Evaluation: {seconds:.2f} s")

    output = OUTPUTS / "evaluation_results.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Algorithm", "Fitness", "Evaluation Time (s)"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved 5 final results: {output.name}")
    print("No optimized rover .npy files were modified.")


if __name__ == "__main__":
    main()
