
import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS

# Genetic Algorithm - ESA Morphing Rovers

rng = np.random.default_rng(42)

POPULATION_SIZE = 6
GENERATIONS = 5
MUTATION_COUNT = 50
MUTATION_STRENGTH = 0.02

# Marrim konfigurimin me te mire nga Adaptive Hill Climbing
initial_rover = np.load("adaptive_best_rover.npy").copy()

# Ndryshojme vetem parametrat e rrjetit nervor
# 4 format e roverit dhe 7 parametrat diskrete mbeten te pandryshuar
mutable_indices = np.arange(
    NUM_MODE_PARAMETERS,
    len(initial_rover) - 7
)


# Funksioni per vleresimin e roverit
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
        child[indices], -100, 100
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


# Krijimi i popullates fillestare
initial_fitness = evaluate(initial_rover)

best_rover = initial_rover.copy()
best_fitness = initial_fitness

population = [(initial_rover.copy(), initial_fitness)]

print(f"Fitness fillestar: {initial_fitness:.4f}")

for i in range(POPULATION_SIZE - 1):
    rover = mutate(initial_rover)
    fitness = evaluate(rover)

    population.append((rover, fitness))

    if fitness < best_fitness:
        best_fitness = fitness
        best_rover = rover.copy()


# Procesi i evolucionit
for generation in range(GENERATIONS):

    # SELECTION - zgjedhim roveret me te mire
    population.sort(key=lambda x: x[1])

    # ELITISM - ruajme roverin me te mire
    new_population = [population[0]]

    # Krijojme pasardhesit
    while len(new_population) < POPULATION_SIZE:

        parent1 = population[0][0]
        parent2 = population[1][0]

        # CROSSOVER
        child = crossover(parent1, parent2)

        # MUTATION
        child = mutate(child)

        # FITNESS
        fitness = evaluate(child)

        new_population.append((child, fitness))

        if fitness < best_fitness:
            best_fitness = fitness
            best_rover = child.copy()

    population = new_population

    print(
        f"Generation {generation + 1}: "
        f"Best Fitness = {best_fitness:.4f}"
    )


# Ruajme konfigurimin me te mire
np.save("ga_best_rover.npy", best_rover)

print("\nGenetic Algorithm completed!")
print(f"Initial Fitness: {initial_fitness:.4f}")
print(f"Best Fitness: {best_fitness:.4f}")
