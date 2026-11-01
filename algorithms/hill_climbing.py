
from pathlib import Path

import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS

best_rover = udp.example().copy()
best_fitness = float(udp.fitness(best_rover)[0])

print("Initial fitness:", best_fitness)

rng = np.random.default_rng(42)

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

    print(f"Test {i + 1}: Fitness = {fitness:.4f}")

    if fitness < best_fitness:
        best_fitness = fitness
        best_rover = candidate.copy()
        print("Better configuration found!")

output = Path("results/hill_climbing_trial.npy")

if output.exists():
    raise SystemExit("Trial result already exists. Nothing overwritten.")

output.parent.mkdir(parents=True, exist_ok=True)
np.save(output, best_rover)

print("\nBest fitness:", best_fitness)
