
import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS

# Genetic Algorithm - ESA Morphing Rovers
 
rng = np.random.default_rng(42)

POPULATION_SIZE = 8
GENERATIONS = 10
MUTATION_COUNT = 50
MUTATION_STRENGTH = 0.02

# Marrim roverin me te mire nga Genetic Algorithm i kaluar
initial_rover = np.load("ga_best_rover.npy").copy()

# Ndryshojme vetem parametrat e rrjetit nervor
# 4 format e roverit dhe 7 parametrat diskrete mbeten te pandryshuar
mutable_indices = np.arange(
    NUM_MODE_PARAMETERS,
    len(initial_rover) - 7
)


# FITNESS - vleresimi i roverit
def evaluate(rover):
    fitness = float(udp.fitness(rover)[0])

    if not np.isfinite(fitness):
        return float("inf")

    return fitness


# MUTATION - ndryshimi i disa parametrave
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


# CROSSOVER - kombinimi i dy prinderve
def crossover(parent1, parent2):
    child = parent1.copy()

    indices = rng.choice(
        mutable_indices,
        size=MUTATION_COUNT,
        replace=False
    )

    child[indices] = parent2[indices]

    return child


# POPULATION - krijimi i popullates fillestare
initial_fitness = evaluate(initial_rover)

best_rover = initial_rover.copy()
best_fitness = initial_fitness

population = [
    (initial_rover.copy(), initial_fitness)
]

print(f"Fitness fillestar: {initial_fitness:.4f}")

# Krijojme 7 roveret e tjere me mutacione
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


# GENETIC ALGORITHM - procesi i evolucionit
for generation in range(GENERATIONS):

    # SELECTION - renditja sipas Fitness
    population.sort(key=lambda x: x[1])

    # ELITISM - ruajme roverin me te mire
    new_population = [
        (population[0][0].copy(), population[0][1])
    ]

    # Krijimi i pasardhesve
    while len(new_population) < POPULATION_SIZE:

        # Zgjedhim dy prinderit me te mire
        parent1 = population[0][0]
        parent2 = population[1][0]

        # CROSSOVER
        child = crossover(parent1, parent2)

        # MUTATION
        child = mutate(child)

        # FITNESS
        fitness = evaluate(child)

        new_population.append((child, fitness))

        # Ruajme konfigurimin me te mire
        if fitness < best_fitness:

            best_fitness = fitness
            best_rover = child.copy()

            print(
                f"Improvement found in "
                f"generation {generation + 1}: "
                f"{best_fitness:.4f}"
            )

    # Kalojme ne gjeneraten e ardhshme
    population = new_population

    print(
        f"Generation {generation + 1}/{GENERATIONS}: "
        f"Best Fitness = {best_fitness:.4f}"
    )


# Ruajme rezultatin ne fajll te ri
np.save("ga_best_rover_v2.npy", best_rover)

print("\nGenetic Algorithm V2 completed!")
print(f"Initial Fitness: {initial_fitness:.4f}")
print(f"Best Fitness: {best_fitness:.4f}")
print("Saved as: ga_best_rover_v2.npy")
