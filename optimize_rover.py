
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

        print(
            f"Iteration {i+1}: "
            f"Best = {best_fitness:.6f}"
        )

    np.save("best_rover.npy", best_rover)
    print(f"\nFinal Fitness: {best_fitness:.6f}")


# ALGORITHM 2 - ADAPTIVE COORDINATE SEARCH

def coordinate_search():

    # Seed i ri per eksperimente te reja
    rng = np.random.default_rng(2038)

    output = "weights_best_rover.npy"

    # Fillojme nga rezultati me i mire
    best_rover = np.load(output).copy()
    best_fitness = float(udp.fitness(best_rover)[0])

    initial_fitness = best_fitness

    lower, upper = udp.get_bounds()

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Kufijte e peshave te rrjetit nervor
    weight_start = NUM_MODE_PARAMETERS
    weight_end = weight_start + NUM_WEIGHTS

    num_selected_weights = 12

    selected_weights = rng.choice(
        np.arange(weight_start, weight_end),
        size=num_selected_weights,
        replace=False
    )

    improvements = 0
    evaluations = 0

    print("\nESA Morphing Rovers")
    print("Adaptive Coordinate Search")
    print("-" * 45)

    print(f"Initial Fitness: {best_fitness:.6f}")
    print(f"Total Weights: {NUM_WEIGHTS}")
    print(f"Selected Weights: {num_selected_weights}")

    print("-" * 45)

    for i, index in enumerate(selected_weights, 1):

        # Hapi fillestar
        delta = float(
            rng.choice([0.001, 0.005, 0.02])
        )

        # Te dy drejtimet duhet te nisen
        # nga i njejti konfigurim
        base_rover = best_rover.copy()

        local_best_rover = None
        local_best_fitness = best_fitness

        print(
            f"\nWeight {i}/{num_selected_weights} "
            f"| Index = {index}"
        )

        # Fillimisht hapi normal,
        # pastaj hapi 5 here me i vogel
        steps = [
            ("Normal", delta),
            ("Refined", delta / 5)
        ]

        for stage, step in steps:

            # Nese hapi normal ka gjetur
            # permiresim, kalojme te pesha tjeter
            if (
                stage == "Refined"
                and local_best_rover is not None
            ):
                break

            for direction in [-1, 1]:

                candidate = base_rover.copy()

                candidate[index] += direction * step

                # Respektojme kufijte e ESA-s
                candidate[index] = np.clip(
                    candidate[index],
                    lower[index],
                    upper[index]
                )

                # Mos vlereso konfigurim identik
                if candidate[index] == base_rover[index]:
                    continue

                fitness = float(
                    udp.fitness(candidate)[0]
                )

                evaluations += 1

                print(
                    f"{stage} | "
                    f"Direction {direction:+d} | "
                    f"Step = {step:.6f} | "
                    f"Fitness = {fitness:.6f}"
                )

                # Mbajme rezultatin me te mire
                # per kete peshe
                if (
                    np.isfinite(fitness)
                    and fitness < local_best_fitness - 1e-6
                ):

                    local_best_fitness = fitness
                    local_best_rover = candidate.copy()

        # Pranojme vetem konfigurimin me te mire,
        # pasi i kemi krahasuar drejtimet
        if local_best_rover is not None:

            best_rover = local_best_rover.copy()
            best_fitness = local_best_fitness

            improvements += 1

            # Ruajme cdo rekord te ri
            np.save(output, best_rover)

            print(
                f"NEW BEST FITNESS: "
                f"{best_fitness:.6f}"
            )

            print(f"Saved: {output}")

        else:

            print("No improvement for this weight.")

        print(
            f"Current Best: {best_fitness:.6f}"
        )

    # FINAL RESULTS

    print("\n" + "=" * 45)
    print("ADAPTIVE COORDINATE SEARCH COMPLETED")
    print("=" * 45)

    print(f"Initial Fitness: {initial_fitness:.6f}")
    print(f"Best Fitness: {best_fitness:.6f}")
    print(f"Improvements: {improvements}")
    print(f"Fitness Evaluations: {evaluations}")

    if improvements > 0:

        print("\nNew best rover saved successfully!")

    else:

        print("\nNo improvement found.")
        print("Previous best rover preserved.")


# ALGORITHM 3 - FOCUSED COORDINATE SEARCH

def focused_coordinate_search():

    output = "weights_best_rover.npy"

    best_rover = np.load(output).copy()
    best_fitness = float(udp.fitness(best_rover)[0])
    initial_fitness = best_fitness

    lower, upper = udp.get_bounds()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Peshat qe kane dhene permiresime
    targets = [
        (3847, 0.0002),
        (4690, 0.0200),
        (6364, 0.0040),
        (14052, 0.0010),
        (2784, 0.0002)
    ]

    improvements = 0
    evaluations = 0

    print("\nESA Morphing Rovers")
    print("Focused Coordinate Search")
    print(f"Initial Fitness: {initial_fitness:.6f}")
    print("-" * 45)

    for i, (index, original_step) in enumerate(targets, 1):

        base_rover = best_rover.copy()

        local_best = best_fitness
        local_candidate = None

        print(f"\nWeight {i}/5 | Index = {index}")

        # Provojme hapa me madhesi te ndryshme
        for scale in [0.5, 1.0, 2.0]:

            step = original_step * scale

            # Krahasojme drejtimet nga e njejta baze
            for direction in [-1, 1]:

                candidate = base_rover.copy()

                candidate[index] = np.clip(
                    candidate[index] + direction * step,
                    lower[index],
                    upper[index]
                )

                if candidate[index] == base_rover[index]:
                    continue

                fitness = float(udp.fitness(candidate)[0])
                evaluations += 1

                print(
                    f"Step = {step:.6f} | "
                    f"Direction {direction:+d} | "
                    f"Fitness = {fitness:.6f}"
                )

                if (
                    np.isfinite(fitness)
                    and fitness < local_best - 1e-6
                ):
                    local_best = fitness
                    local_candidate = candidate.copy()

        if local_candidate is not None:

            best_rover = local_candidate
            best_fitness = local_best
            improvements += 1

            # Ruajme vetem konfigurimin me te mire
            np.save(output, best_rover)

            print(
                f"NEW BEST FITNESS: {best_fitness:.6f}"
            )
            print(f"Saved: {output}")

        else:
            print("No improvement for this weight.")

    print("\n" + "=" * 45)
    print("FOCUSED SEARCH COMPLETED")
    print("=" * 45)

    print(f"Initial Fitness: {initial_fitness:.6f}")
    print(f"Best Fitness: {best_fitness:.6f}")
    print(f"Improvements: {improvements}")
    print(f"Fitness Evaluations: {evaluations}")

    if improvements == 0:
        print("Previous best rover preserved.")
    else:
        print("New best rover saved successfully!")



# SELECT ALGORITHM

if __name__ == "__main__":

    if len(sys.argv) == 2 and sys.argv[1] == "hill":
        hill_climbing()

    elif len(sys.argv) == 2 and sys.argv[1] == "coordinate":
        coordinate_search()

    elif len(sys.argv) == 2 and sys.argv[1] == "focused":
        focused_coordinate_search()

    else:
        print("Usage:")
        print("python optimize_rover.py hill")
        print("python optimize_rover.py coordinate")
        print("python optimize_rover.py focused")
