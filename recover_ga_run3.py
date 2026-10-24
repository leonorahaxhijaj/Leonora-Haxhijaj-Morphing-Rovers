
import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS

# Riprodhimi i Run 3 te Genetic Algorithm
# Seed i njejte me compare_algorithms.py

rng = np.random.default_rng(2028)

EVALUATIONS = 6
POPULATION_SIZE = 4
MUTATION_COUNT = 50

baseline = udp.example().copy()

mutable_indices = np.arange(
    NUM_MODE_PARAMETERS,
    len(baseline) - 7
)


def evaluate(rover):
    fitness = float(udp.fitness(rover)[0])
    return fitness if np.isfinite(fitness) else float("inf")


def mutate(rover, sigma=0.02):
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


def crossover(parent1, parent2):
    child = parent1.copy()

    indices = rng.choice(
        mutable_indices,
        size=MUTATION_COUNT,
        replace=False
    )

    child[indices] = parent2[indices]

    return child


# Konfigurimi fillestar
baseline_fitness = evaluate(baseline)

best_rover = baseline.copy()
best_fitness = baseline_fitness

population = [
    (baseline.copy(), baseline_fitness)
]

print(f"ESA Baseline: {baseline_fitness:.4f}")


# Riprodhojme 6 vleresimet e Run 3
for i in range(EVALUATIONS):

    if len(population) < POPULATION_SIZE:

        candidate = mutate(baseline)

    else:

        population.sort(key=lambda x: x[1])

        parent1 = population[0][0]
        parent2 = population[1][0]

        candidate = crossover(parent1, parent2)
        candidate = mutate(candidate)

    fitness = evaluate(candidate)

    population.append(
        (candidate.copy(), fitness)
    )

    population.sort(key=lambda x: x[1])
    population = population[:POPULATION_SIZE]

    if fitness < best_fitness:
        best_fitness = fitness
        best_rover = candidate.copy()

    print(
        f"Evaluation {i + 1}: "
        f"Fitness = {fitness:.4f}, "
        f"Best = {best_fitness:.4f}"
    )


# Kontrollojme nese kemi riprodhuar rezultatin
if abs(best_fitness - 6.7131) < 0.0001:

    np.save("ga_best_rover_run3.npy", best_rover)

    print("\nRekordi u riprodhua me sukses!")
    print("Saved: ga_best_rover_run3.npy")

else:
    print("\nRezultati nuk perputhet me Run 3.")
    print("Kontrollo konfigurimin dhe kodin origjinal.")

print(f"Best Fitness: {best_fitness:.4f}")
