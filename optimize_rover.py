
import sys
import numpy as np

from morphing_udp import udp, NUM_MODE_PARAMETERS, NUM_WEIGHTS


# ALGORITHM 1 - HILL CLIMBING
def hill_climbing():

    rng = np.random.default_rng(42)

    best_rover = udp.example().copy()
    best_fitness = float(udp.fitness(best_rover)[0])

    print(f"Initial Fitness: {best_fitness:.6f}")

    for i in range(10):

        candidate = best_rover.copy()

        indices = rng.choice(
            np.arange(NUM_MODE_PARAMETERS, len(candidate) - 7),
            size=30,
            replace=False
        )

        candidate[indices] += rng.normal(0, 0.02, 30)
        candidate[indices] = np.clip(
            candidate[indices], -100, 100
        )

        fitness = float(udp.fitness(candidate)[0])

        if np.isfinite(fitness) and fitness < best_fitness:
            best_fitness = fitness
            best_rover = candidate.copy()

        print(f"Iteration {i+1}: Best = {best_fitness:.6f}")

    np.save("best_rover.npy", best_rover)
    print(f"\nFinal Fitness: {best_fitness:.6f}")


# ALGORITHM 2 - COORDINATE SEARCH
def coordinate_search():

    rng = np.random.default_rng(2036)

    best_rover = np.load("ga_best_rover_new.npy").copy()
    best_fitness = float(udp.fitness(best_rover)[0])

    lower, upper = udp.get_bounds()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    weight_start = NUM_MODE_PARAMETERS
    weight_end = weight_start + NUM_WEIGHTS

    print(f"Initial Fitness: {best_fitness:.6f}")

    selected_weights = rng.choice(
        np.arange(weight_start, weight_end),
        size=8,
        replace=False
    )

    improvements = 0

    for i, index in enumerate(selected_weights, 1):

        delta = float(rng.choice([0.001, 0.005, 0.02]))

        for direction in [-1, 1]:

            candidate = best_rover.copy()

            candidate[index] += direction * delta

            candidate[index] = np.clip(
                candidate[index],
                lower[index],
                upper[index]
            )

            if candidate[index] == best_rover[index]:
                continue

            fitness = float(udp.fitness(candidate)[0])

            if (
                np.isfinite(fitness)
                and fitness < best_fitness - 1e-6
            ):
                best_fitness = fitness
                best_rover = candidate.copy()
                improvements += 1

                print("New improvement found!")

            print(
                f"Weight {i}/8 | "
                f"Direction {direction:+d} | "
                f"Fitness = {fitness:.6f} | "
                f"Best = {best_fitness:.6f}"
            )

    # Nuk mbishkruajme nje rover ekzistues me te mire
    import os

    output = "weights_best_rover.npy"
    saved_fitness = float("inf")

    if os.path.exists(output):
        saved = np.load(output)
        saved_fitness = float(udp.fitness(saved)[0])

    if best_fitness < saved_fitness - 1e-6:
        np.save(output, best_rover)
        print(f"Saved: {output}")

    print("\nCoordinate Search completed!")
    print(f"Best Fitness: {best_fitness:.6f}")
    print(f"Improvements: {improvements}")


# Zgjedhja e algoritmit
if __name__ == "__main__":

    if len(sys.argv) == 2 and sys.argv[1] == "hill":
        hill_climbing()

    elif len(sys.argv) == 2 and sys.argv[1] == "coordinate":
        coordinate_search()

    else:
        print("Usage:")
        print("python optimize_rover.py hill")
        print("python optimize_rover.py coordinate")
