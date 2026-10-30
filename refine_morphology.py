
import os
import numpy as np
from morphing_udp import udp

SOURCE = "multistart_ga_best_rover.npy"
OUTPUT = "refined_morphology_best_rover.npy"

INDICES = [22, 204]
STEPS = [0.02, 0.005, 0.001, 0.0002]

if os.path.exists(OUTPUT):
    raise SystemExit("Output already exists. File preserved.")

best = np.load(SOURCE, allow_pickle=False).copy()
initial = float(udp.fitness(best)[0])
best_fitness = initial

lower, upper = udp.get_bounds()
lower = np.asarray(lower, dtype=float)
upper = np.asarray(upper, dtype=float)

assert len(best) == len(lower) == len(upper)
assert np.all(np.isfinite(best))
assert np.all(best >= lower)
assert np.all(best <= upper)

evaluations = 0
improvements = 0

print("JOINT MORPHOLOGY REFINEMENT")
print("=" * 50)
print(f"Initial Fitness: {initial:.9f}")
print(f"Optimized Indices: {INDICES}")

for step in STEPS:

    print(f"\nSTEP SIZE: {step}")

    base = best.copy()
    local_best = best_fitness
    local_candidate = None

    for direction1 in [-1, 0, 1]:
        for direction2 in [-1, 0, 1]:

            if direction1 == 0 and direction2 == 0:
                continue

            candidate = base.copy()

            candidate[22] = np.clip(
                base[22] + direction1 * step,
                lower[22],
                upper[22]
            )

            candidate[204] = np.clip(
                base[204] + direction2 * step,
                lower[204],
                upper[204]
            )

            if np.array_equal(candidate, base):
                continue

            fitness = float(udp.fitness(candidate)[0])
            evaluations += 1

            print(
                f"Directions ({direction1:+d}, {direction2:+d})"
                f" | Fitness: {fitness:.9f}"
            )

            if (
                np.isfinite(fitness)
                and fitness < local_best - 1e-6
            ):
                local_best = fitness
                local_candidate = candidate.copy()

    if local_candidate is not None:

        best = local_candidate
        best_fitness = local_best
        improvements += 1

        np.save(OUTPUT, best)

        print(f"NEW BEST FITNESS: {best_fitness:.9f}")
    else:
        print("No improvement at this step size.")

print("\n" + "=" * 50)
print("JOINT MORPHOLOGY REFINEMENT COMPLETED")
print("=" * 50)

print(f"Initial Fitness: {initial:.9f}")
print(f"Best Fitness: {best_fitness:.9f}")
print(f"Improvements: {improvements}")
print(f"Fitness Evaluations: {evaluations}")
print(f"Total Improvement: {initial - best_fitness:.9f}")

if improvements > 0:
    print(f"Saved: {OUTPUT}")
else:
    print("No improvement. No new file saved.")

print("Original rover preserved.")
