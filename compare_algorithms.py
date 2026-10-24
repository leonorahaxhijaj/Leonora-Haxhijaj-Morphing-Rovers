
import csv
import time
import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS

# ESA Morphing Rovers - Fair Algorithm Comparison

EVALUATIONS = 6
RUNS = 3
MUTATION_COUNT = 50
POPULATION_SIZE = 4

baseline = udp.example().copy()
baseline_fitness = float(udp.fitness(baseline)[0])

mutable_indices = np.arange(
    NUM_MODE_PARAMETERS,
    len(baseline) - 7
)


def evaluate(rover):
    value = float(udp.fitness(rover)[0])
    return value if np.isfinite(value) else float("inf")


def mutate(rover, rng, sigma):
    child = rover.copy()

    indices = rng.choice(
        mutable_indices,
        size=MUTATION_COUNT,
        replace=False
    )

    child[indices] += rng.normal(
        0, sigma, MUTATION_COUNT
    )

    child[indices] = np.clip(
        child[indices], -100, 100
    )

    return child


def crossover(parent1, parent2, rng):
    child = parent1.copy()

    indices = rng.choice(
        mutable_indices,
        size=MUTATION_COUNT,
        replace=False
    )

    child[indices] = parent2[indices]

    return child


def run_algorithm(method, seed):
    rng = np.random.default_rng(seed)

    best_rover = baseline.copy()
    best_fitness = baseline_fitness

    sigma = 0.02

    # Popullata perdoret vetem nga Genetic Algorithm
    population = [
        (baseline.copy(), baseline_fitness)
    ]

    start = time.perf_counter()

    for i in range(EVALUATIONS):

        if method == "Genetic Algorithm":

            if len(population) < POPULATION_SIZE:
                candidate = mutate(
                    baseline, rng, 0.02
                )

            else:
                population.sort(key=lambda x: x[1])

                parent1 = population[0][0]
                parent2 = population[1][0]

                candidate = crossover(
                    parent1, parent2, rng
                )

                candidate = mutate(
                    candidate, rng, 0.02
                )

        else:
            candidate = mutate(
                best_rover, rng, sigma
            )

        fitness = evaluate(candidate)

        if method == "Genetic Algorithm":
            population.append(
                (candidate.copy(), fitness)
            )

            population.sort(key=lambda x: x[1])
            population = population[:POPULATION_SIZE]

        if fitness < best_fitness:
            best_fitness = fitness
            best_rover = candidate.copy()
            improved = True
        else:
            improved = False

        if method == "Adaptive Hill Climbing":
            if improved:
                sigma = min(sigma * 1.2, 0.1)
            else:
                sigma = max(sigma * 0.85, 0.002)

    elapsed = time.perf_counter() - start

    return best_fitness, elapsed


algorithms = [
    "Hill Climbing",
    "Adaptive Hill Climbing",
    "Genetic Algorithm"
]

results = []

print(f"ESA baseline: {baseline_fitness:.4f}")
print(f"Evaluations per algorithm: {EVALUATIONS}")

for run in range(RUNS):
    print(f"\nRun {run + 1}/{RUNS}")

    for method in algorithms:
        fitness, elapsed = run_algorithm(
            method, seed=2026 + run
        )

        results.append({
            "Run": run + 1,
            "Algorithm": method,
            "Fitness": fitness,
            "Optimization Time (s)": round(elapsed, 2),
            "Evaluations": EVALUATIONS
        })

        print(
            f"{method}: Fitness = {fitness:.4f}, "
            f"Time = {elapsed:.2f}s"
        )


with open(
    "comparison_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Run",
            "Algorithm",
            "Fitness",
            "Optimization Time (s)",
            "Evaluations"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nComparison completed!")
print("Saved: comparison_results.csv")
