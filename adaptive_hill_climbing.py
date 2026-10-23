
import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS

# Marrim roverin me te mire nga eksperimenti i kaluar
best_rover = np.load("best_rover.npy").copy()
best_fitness = float(udp.fitness(best_rover)[0])

print(f"Fitness fillestar: {best_fitness:.4f}")

rng = np.random.default_rng(123)

# Parametrat fillestare te algoritmit
sigma = 0.02
num_iterations = 10
num_mutations = 30

for i in range(num_iterations):

    candidate = best_rover.copy()

    # Zgjedhim rastesisht 30 parametra te rrjetit nervor
    indices = rng.choice(
        np.arange(NUM_MODE_PARAMETERS, len(candidate) - 7),
        size=num_mutations,
        replace=False
    )

    # Ndryshojme parametrat sipas vleres sigma
    candidate[indices] += rng.normal(
        0, sigma, num_mutations
    )

    candidate[indices] = np.clip(
        candidate[indices], -100, 100
    )

    # Vleresojme konfigurimin e ri
    fitness = float(udp.fitness(candidate)[0])

    if fitness < best_fitness:
        best_fitness = fitness
        best_rover = candidate.copy()

        # Rrisim madhesine e ndryshimeve
        sigma = min(sigma * 1.2, 0.1)
        print("U gjet konfigurim me i mire!")
    else:
        # Zvogelojme madhesine e ndryshimeve
        sigma = max(sigma * 0.85, 0.002)

    print(
        f"Iteracioni {i+1}: "
        f"Fitness = {fitness:.4f}, "
        f"Best = {best_fitness:.4f}, "
        f"Sigma = {sigma:.5f}"
    )

# Ruajme roverin me te mire
np.save("adaptive_best_rover.npy", best_rover)

print("\nFitness perfundimtar:", best_fitness)
