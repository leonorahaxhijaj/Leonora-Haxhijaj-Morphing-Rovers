
import numpy as np
from morphing_udp import udp, NUM_MODE_PARAMETERS

# Konfigurimi fillestar
best_rover = udp.example().copy()
best_fitness = float(udp.fitness(best_rover)[0])

print("Fitness fillestar:", best_fitness)

rng = np.random.default_rng(42)

# Provojme 10 konfigurime
for i in range(10):
    candidate = best_rover.copy()

    # Zgjedhim 30 parametra rastesisht
    indices = rng.choice(
        np.arange(NUM_MODE_PARAMETERS, len(candidate) - 7),
        size=30,
        replace=False
    )

    # Bejme ndryshime te vogla
    candidate[indices] += rng.normal(0, 0.02, 30)

    candidate[indices] = np.clip(
        candidate[indices], -100, 100
    )

    # Llogarisim fitness
    fitness = float(udp.fitness(candidate)[0])

    print(f"Testi {i + 1}: Fitness = {fitness:.4f}")

    if fitness < best_fitness:
        best_fitness = fitness
        best_rover = candidate.copy()
        print("U gjet nje konfigurim me i mire!")

# Ruajme konfigurimin me te mire
np.save("best_rover.npy", best_rover)

print("\nFitness me i mire:", best_fitness)
