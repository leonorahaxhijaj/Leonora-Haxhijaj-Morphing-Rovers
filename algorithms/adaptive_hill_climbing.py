
from pathlib import Path

OUTPUT_PATH = Path("results/adaptive_hill_climbing_trial.npy")

if OUTPUT_PATH.exists():
    raise SystemExit("Trial result already exists. Nothing overwritten.")


import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS

best_rover = np.load("final_results/hill_climbing_best_rover.npy", allow_pickle=False).copy()
best_fitness = float(udp.fitness(best_rover)[0])

print(f"Initial fitness: {best_fitness:.4f}")

rng = np.random.default_rng(123)

sigma = 0.02
num_iterations = 10
num_mutations = 30

for i in range(num_iterations):

    candidate = best_rover.copy()

    indices = rng.choice(
        np.arange(NUM_MODE_PARAMETERS, len(candidate) - 7),
        size=num_mutations,
        replace=False
    )

    candidate[indices] += rng.normal(
        0, sigma, num_mutations
    )

    candidate[indices] = np.clip(
        candidate[indices], -100, 100
    )

    fitness = float(udp.fitness(candidate)[0])

    if fitness < best_fitness:
        best_fitness = fitness
        best_rover = candidate.copy()

        sigma = min(sigma * 1.2, 0.1)
        print("Better configuration found!")
    else:
        sigma = max(sigma * 0.85, 0.002)

    print(
        f"Iteration {i+1}: "
        f"Fitness = {fitness:.4f}, "
        f"Best = {best_fitness:.4f}, "
        f"Sigma = {sigma:.5f}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_PATH, best_rover)

print("\nFinal fitness:", best_fitness)
