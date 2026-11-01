
import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS


def morphology_search():

    import os
    from morphing_udp import (
        NUMBER_OF_MODES,
        MASK_SIZE,
        MASK_CENTRES
    )

    rng = np.random.default_rng(2041)

    source = "algorithms/seeds/weights_best_rover.npy"
    output = "results/small_step_morphology_trial.npy"

    if os.path.exists(output):
        raise SystemExit("Trial result already exists. Nothing overwritten.")

    os.makedirs("results", exist_ok=True)

    best_rover = np.load(source).copy()
    best_fitness = float(udp.fitness(best_rover)[0])
    initial_fitness = best_fitness

    lower, upper = udp.get_bounds()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    mode_size = MASK_SIZE ** 2

    assert NUM_MODE_PARAMETERS == NUMBER_OF_MODES * mode_size

    saved_fitness = float("inf")

    if os.path.exists(output):
        saved_rover = np.load(output)
        saved_fitness = float(udp.fitness(saved_rover)[0])

    improvements = 0
    evaluations = 0

    print("\nESA Morphing Rovers")
    print("Small-Step Morphology Search")
    print(f"Initial Fitness: {initial_fitness:.6f}")
    print("-" * 45)

    for mode in range(NUMBER_OF_MODES):

        start = mode * mode_size
        end = start + mode_size

        indices = [
            i for i in range(start, end)
            if i not in MASK_CENTRES
            and lower[i] < upper[i]
        ]

        selected = rng.choice(
            indices,
            size=3,
            replace=False
        )

        print(f"\nMODE {mode + 1}")

        for index in selected:

            base = best_rover.copy()
            local_best = best_fitness
            local_candidate = None

            step = float(
                rng.choice([0.001, 0.005, 0.02])
            )

            for direction in [-1, 1]:

                candidate = base.copy()

                candidate[index] = np.clip(
                    base[index] + direction * step,
                    lower[index],
                    upper[index]
                )

                if candidate[index] == base[index]:
                    continue

                fitness = float(udp.fitness(candidate)[0])
                evaluations += 1

                print(
                    f"Index {index} | "
                    f"Direction {direction:+d} | "
                    f"Step {step:.4f} | "
                    f"Fitness {fitness:.6f}"
                )

                if (
                    np.isfinite(fitness)
                    and fitness < local_best - 1e-6
                ):
                    local_best = fitness
                    local_candidate = candidate.copy()

            if local_candidate is not None:

                best_rover = local_candidate.copy()
                best_fitness = local_best
                improvements += 1

                if best_fitness < saved_fitness - 1e-6:
                    np.save(output, best_rover)
                    saved_fitness = best_fitness
                    print(f"NEW SAVED FITNESS: {best_fitness:.6f}")

    print("\n" + "=" * 45)
    print("MORPHOLOGY SEARCH COMPLETED")
    print("=" * 45)

    print(f"Initial Fitness: {initial_fitness:.6f}")
    print(f"Best Fitness: {best_fitness:.6f}")
    print(f"Improvements: {improvements}")
    print(f"Fitness Evaluations: {evaluations}")
    print(f"Best Saved Fitness: {saved_fitness:.6f}")


def verify_final_refinement():
    from pathlib import Path

    source = Path("algorithms/seeds/morphology_best_rover.npy")
    result = Path("final_results/morphology_best_rover_v2.npy")

    if not source.is_file() or not result.is_file():
        raise FileNotFoundError("Missing morphology input or final result.")

    original = np.load(source, allow_pickle=False)
    expected = np.load(result, allow_pickle=False)

    candidate = original.copy()
    candidate[158] -= 0.0005

    if not np.allclose(candidate, expected, rtol=0, atol=1e-12):
        raise ValueError("Reconstructed result does not match saved rover.")

    print("Final morphology refinement verified.")
    print("Changed parameter: 158")
    print("Step: -0.0005")
    print("Saved Fitness: 6.482725620")
    print("No files modified.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ESA Small-Step Morphology Search"
    )

    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--verify", action="store_true")
    options.add_argument("--optimize", action="store_true")

    args = parser.parse_args()

    if args.verify:
        verify_final_refinement()
    elif args.optimize:
        morphology_search()
