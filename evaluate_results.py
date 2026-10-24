
import csv
import time
import numpy as np
from morphing_udp import udp

# Konfigurimet qe do t'i vleresojme
rovers = {
    "ESA Baseline": udp.example(),
    "Hill Climbing": np.load("best_rover.npy"),
    "Adaptive Hill Climbing": np.load("adaptive_best_rover.npy"),
    "Genetic Algorithm V1": np.load("ga_best_rover.npy"),
    "Genetic Algorithm V2": np.load("ga_best_rover_v2.npy")
}

results = []

print("ESA Morphing Rovers - Results Evaluation\n")

for name, rover in rovers.items():

    start = time.perf_counter()

    fitness = float(udp.fitness(rover)[0])

    elapsed = time.perf_counter() - start

    results.append({
        "Algorithm": name,
        "Fitness": fitness,
        "Evaluation Time (s)": elapsed
    })

    print(
        f"{name}: Fitness = {fitness:.4f}, "
        f"Evaluation Time = {elapsed:.2f}s"
    )

# Ruajme rezultatet ne CSV
with open("evaluation_results.csv", "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Algorithm",
            "Fitness",
            "Evaluation Time (s)"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to evaluation_results.csv")
