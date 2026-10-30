import os
import numpy as np

from morphing_udp import (
    udp,
    NUM_MODE_PARAMETERS,
    NUM_BIASES,
    NUM_WEIGHTS,
    MASK_CENTRES,
)

rng = np.random.default_rng(2048)

SOURCE = "adaptive_local_best_rover.npy"
OUTPUT = "multistart_ga_best_rover.npy"

POP_SIZE = 6
GENERATIONS = 3
SCALES = [0.4, 1.0, 2.0]

if os.path.exists(OUTPUT):
    raise SystemExit(
        f"Output '{OUTPUT}' already exists. "
        "Existing result preserved. Choose a new OUTPUT name to rerun."
    )

lower, upper = udp.get_bounds()
lower = np.asarray(lower, dtype=float)
upper = np.asarray(upper, dtype=float)

best = np.load(SOURCE, allow_pickle=False).copy()
assert best.ndim == 1
assert len(best) == len(lower) == len(upper)
assert np.all(np.isfinite(best))
assert np.all(best >= lower)
assert np.all(best <= upper)

initial_fitness = float(udp.fitness(best)[0])
best_fitness = initial_fitness

evaluations = 1
improvements = 0

weight_start = NUM_MODE_PARAMETERS + NUM_BIASES
output_start = weight_start + NUM_WEIGHTS - 80
variable_end = weight_start + NUM_WEIGHTS

centres = set(MASK_CENTRES)

morph_indices = np.array([
    i for i in range(NUM_MODE_PARAMETERS)
    if i not in centres
])

hot_indices = np.array([61, 70, 95, 158, 204])


def mutate(x, scale):
    """Apply a small random mutation to morphology, weights, or biases."""
    candidate = x.copy()

    kind = rng.choice(
        ["morphology", "network", "bias", "mixed"],
        p=[0.35, 0.40, 0.10, 0.15],
    )

    if kind in ("morphology", "mixed"):
        count = int(rng.choice([1, 2, 4, 8]))
        indices = rng.choice(morph_indices, size=count, replace=False)

        if rng.random() < 0.25:
            indices[0] = int(rng.choice(hot_indices))

        step = float(rng.choice([0.0002, 0.001, 0.003, 0.01])) * scale
        candidate[indices] += rng.normal(0, step, size=count)

    if kind in ("network", "mixed"):
        count = int(rng.choice([1, 2, 5, 12]))
        indices = rng.choice(
            np.arange(weight_start, output_start),
            size=count,
            replace=False,
        )
        step = float(rng.choice([0.00005, 0.0002, 0.001, 0.003])) * scale
        candidate[indices] += rng.normal(0, step, size=count)

    if kind == "bias":
        # Hidden-layer biases only; exclude the two output biases.
        count = int(rng.choice([1, 2, 4]))
        indices = rng.choice(
            np.arange(
                NUM_MODE_PARAMETERS,
                NUM_MODE_PARAMETERS + NUM_BIASES - 2,
            ),
            size=count,
            replace=False,
        )
        step = float(rng.choice([0.00001, 0.0001, 0.001])) * scale
        candidate[indices] += rng.normal(0, step, size=count)

    return np.clip(candidate, lower, upper)


def crossover(a, b):
    """Copy a subset of different parameters from donor b into a."""
    child = a.copy()
    differences = np.flatnonzero(
        np.abs(a[:variable_end] - b[:variable_end]) > 1e-12
    )

    if len(differences) == 0:
        return child

    count = min(len(differences), int(rng.choice([1, 3, 10, 25])))
    indices = rng.choice(differences, size=count, replace=False)
    child[indices] = b[indices]
    return child


def evaluate(x):
    """Evaluate across all ESA scenarios; save only a genuine new best."""
    global best, best_fitness, evaluations, improvements

    if not np.all(np.isfinite(x)):
        return float("inf")

    fitness = float(udp.fitness(x)[0])
    evaluations += 1

    if not np.isfinite(fitness):
        return float("inf")

    if fitness < best_fitness - 1e-6:
        best = x.copy()
        best_fitness = fitness
        improvements += 1
        np.save(OUTPUT, best)

        print(f"\n*** NEW BEST FITNESS: {best_fitness:.9f} ***")
        print(f"Checkpoint saved: {OUTPUT}")

    return fitness


# Use the best existing solutions as different starting points.
seed_paths = [
    SOURCE,
    "morphology_best_rover_v2.npy",
    "ga_best_rover_new.npy",
    "weights_best_rover.npy",
]

seeds = []
for path in seed_paths:
    if not os.path.isfile(path):
        continue

    try:
        candidate = np.load(path, allow_pickle=False).copy()
        if candidate.shape != best.shape:
            continue
        if not np.all(np.isfinite(candidate)):
            continue
        if not np.all((candidate >= lower) & (candidate <= upper)):
            continue
        seeds.append((path, candidate))
    except (OSError, ValueError):
        continue

    if len(seeds) == 3:
        break

while len(seeds) < 3:
    seeds.append((SOURCE + " (restart)", best.copy()))

print("\nMULTI-START GENETIC ALGORITHM")
print("=" * 55)
print(f"Initial Fitness: {initial_fitness:.9f}")
print(f"Population Size: {POP_SIZE}")
print(f"Generations per Start: {GENERATIONS}")
print(f"Number of Starts: {len(seeds)}")
print("-" * 55)

interrupted = False

try:
    for start_id, (seed_name, seed) in enumerate(seeds):
        scale = SCALES[start_id]

        print(f"\nSTART {start_id + 1}/3")
        print(f"Seed: {seed_name}")
        print(f"Mutation Scale: {scale}")

        seed_fitness = evaluate(seed)
        population = [(seed_fitness, seed.copy())]

        # Build initial population: seed + five mutations.
        for _ in range(POP_SIZE - 1):
            candidate = mutate(seed, scale)
            fitness = evaluate(candidate)
            population.append((fitness, candidate))

        population.sort(key=lambda item: item[0])
        print(f"Initial population best: {population[0][0]:.9f}")

        for generation in range(1, GENERATIONS + 1):
            population.sort(key=lambda item: item[0])

            # Elitism: preserve the best two from the current generation.
            new_population = population[:2].copy()
            parents = population[:4]

            while len(new_population) < POP_SIZE:
                parent_id = int(rng.integers(0, len(parents)))
                parent = parents[parent_id][1]

                if rng.random() < 0.45:
                    donor_id = int(rng.integers(0, len(parents)))
                    donor = parents[donor_id][1]
                    child = crossover(parent, donor)
                else:
                    child = parent.copy()

                if rng.random() < 0.15:
                    child = best.copy()

                child = mutate(child, scale)
                fitness = evaluate(child)
                new_population.append((fitness, child))

            population = new_population
            population.sort(key=lambda item: item[0])

            print(
                f"Generation {generation}/{GENERATIONS}"
                f" | Population Best: {population[0][0]:.9f}"
                f" | Global Best: {best_fitness:.9f}"
                f" | Evaluations: {evaluations}"
            )

except KeyboardInterrupt:
    interrupted = True
    print("\nSearch interrupted by user.")
    print("Any improved checkpoint has been preserved.")

print("\n" + "=" * 55)
print(
    "MULTI-START GA INTERRUPTED" if interrupted
    else "MULTI-START GA COMPLETED"
)
print("=" * 55)
print(f"Initial Fitness: {initial_fitness:.9f}")
print(f"Best Fitness: {best_fitness:.9f}")
print(f"Improvements: {improvements}")
print(f"Total Evaluations: {evaluations}")
print(f"Total Improvement: {initial_fitness - best_fitness:.9f}")

if improvements > 0:
    print(f"Saved: {OUTPUT}")
else:
    print("No improvement. No new file saved.")

print("Original rover preserved.")
