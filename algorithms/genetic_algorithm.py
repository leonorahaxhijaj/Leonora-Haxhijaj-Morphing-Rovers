
from pathlib import Path

OUTPUT_PATH = Path("results/genetic_algorithm_trial.npy")

if OUTPUT_PATH.exists():
    raise SystemExit("Trial result already exists. Nothing overwritten.")


import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS
 
rng = np.random.default_rng(42)

POPULATION_SIZE = 8
GENERATIONS = 10
MUTATION_COUNT = 50
MUTATION_STRENGTH = 0.02

initial_rover = np.load("algorithms/seeds/ga_best_rover_run3.npy", allow_pickle=False).copy()

mutable_indices = np.arange(
    NUM_MODE_PARAMETERS,
    len(initial_rover) - 7
)

def evaluate(rover):
    fitness = float(udp.fitness(rover)[0])

    if not np.isfinite(fitness):
        return float("inf")

    return fitness

def mutate(rover):
    child = rover.copy()

    indices = rng.choice(
        mutable_indices,
        size=MUTATION_COUNT,
        replace=False
    )

    child[indices] += rng.normal(
        0,
        MUTATION_STRENGTH,
        MUTATION_COUNT
    )

    child[indices] = np.clip(
        child[indices],
        -100,
        100
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

initial_fitness = evaluate(initial_rover)

best_rover = initial_rover.copy()
best_fitness = initial_fitness

population = [
    (initial_rover.copy(), initial_fitness)
]

print(f"Initial fitness: {initial_fitness:.4f}")

for i in range(POPULATION_SIZE - 1):

    rover = mutate(initial_rover)
    fitness = evaluate(rover)

    population.append((rover, fitness))

    if fitness < best_fitness:
        best_fitness = fitness
        best_rover = rover.copy()

        print(
            f"Initial population improvement: "
            f"{best_fitness:.4f}"
        )


for generation in range(GENERATIONS):

    population.sort(key=lambda x: x[1])

    new_population = [
        (population[0][0].copy(), population[0][1])
    ]

    while len(new_population) < POPULATION_SIZE:

        parent1 = population[0][0]
        parent2 = population[1][0]

        child = crossover(parent1, parent2)

        child = mutate(child)

        fitness = evaluate(child)

        new_population.append((child, fitness))

        if fitness < best_fitness:

            best_fitness = fitness
            best_rover = child.copy()

            print(
                f"Improvement found in "
                f"generation {generation + 1}: "
                f"{best_fitness:.4f}"
            )

    population = new_population

    print(
        f"Generation {generation + 1}/{GENERATIONS}: "
        f"Best Fitness = {best_fitness:.4f}"
    )


if best_fitness < initial_fitness - 1e-8:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, best_rover)
    print("New best rover saved!")
else:
    print("No improvement. No new file saved.")

print("\nGenetic Algorithm V2 completed!")
print(f"Initial Fitness: {initial_fitness:.4f}")
print(f"Best Fitness: {best_fitness:.4f}")
print(f"Best trial file: {OUTPUT_PATH} (if improved)")
