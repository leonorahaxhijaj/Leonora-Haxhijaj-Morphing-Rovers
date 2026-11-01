
import os
import hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from morphing_udp import (
    udp,
    NUM_MODE_PARAMETERS,
    NUM_BIASES,
    NUM_WEIGHTS,
    MASK_CENTRES,
)

SOURCE = "algorithms/seeds/multistart_ga_best_rover.npy"
OUTPUT = "results/parallel_de_trial.npy"

POP_SIZE = 40
GENERATIONS = 100
CR = 0.8
F = 0.65
RANDOM_SEED = 2048
MAX_WORKERS = os.cpu_count()

def _evaluate_worker(candidate):
    return float(udp.fitness(candidate)[0])

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    if os.path.exists(OUTPUT):
        raise SystemExit(
            f"'{OUTPUT}' already exists. File preserved; choose a new OUTPUT name."
        )
    if not os.path.isfile(SOURCE):
        raise SystemExit(f"Source rover not found: {SOURCE}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    lower, upper = udp.get_bounds()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    integer_count = int(udp.get_nix())

    best = np.load(SOURCE, allow_pickle=False).copy()

    def valid(x):
        return (
            x.ndim == 1
            and x.shape == lower.shape == upper.shape
            and np.all(np.isfinite(x))
            and np.all((x >= lower) & (x <= upper))
            and np.array_equal(x[-integer_count:], np.trunc(x[-integer_count:]))
        )

    if not valid(best):
        raise SystemExit("The source rover has invalid dimensions or bounds.")

    weight_start = NUM_MODE_PARAMETERS + NUM_BIASES
    output_weight_start = weight_start + NUM_WEIGHTS - 80
    if not (NUM_MODE_PARAMETERS < weight_start < output_weight_start <= len(best)):
        raise SystemExit("Unexpected neural-network parameter layout.")

    centers = set(MASK_CENTRES)

    fitness_cache = {}
    evaluations = 0
    cache_hits = 0
    improvements = 0

    initial_fitness = float(udp.fitness(best)[0])
    if not np.isfinite(initial_fitness):
        raise SystemExit("The source rover has non-finite fitness.")

    best_fitness = initial_fitness
    fitness_cache[hashlib.sha256(best.tobytes()).digest()] = best_fitness
    evaluations += 1

    def get_cache_key(arr):
        return hashlib.sha256(arr.tobytes()).digest()

    def evaluate_batch(candidates, executor):
        nonlocal evaluations, cache_hits

        results = [None] * len(candidates)
        to_eval_indices = []
        to_eval_candidates = []

        for idx, cand in enumerate(candidates):
            if not valid(cand):
                results[idx] = float("inf")
                continue

            key = get_cache_key(cand)
            if key in fitness_cache:
                results[idx] = fitness_cache[key]
                cache_hits += 1
            else:
                to_eval_indices.append(idx)
                to_eval_candidates.append(cand)

        if to_eval_candidates:
            chunk_size = max(1, len(to_eval_candidates) // MAX_WORKERS)
            eval_results = list(executor.map(_evaluate_worker, to_eval_candidates, chunksize=chunk_size))

            for idx, cand, fit in zip(to_eval_indices, to_eval_candidates, eval_results):
                fit_val = fit if np.isfinite(fit) else float("inf")
                key = get_cache_key(cand)
                fitness_cache[key] = fit_val
                results[idx] = fit_val
                evaluations += 1

        return results

    def mutate_de(pop, target_idx, best_idx):
        pop_len = len(pop)
        available = [i for i in range(pop_len) if i != target_idx]
        r1, r2 = rng.choice(available, size=2, replace=False)

        x_best = pop[best_idx]
        x_1 = pop[r1]
        x_2 = pop[r2]

        mutant = x_best + F * (x_1 - x_2)

        dim = len(x_best)
        cross_mask = rng.random(dim) < CR
        j_rand = rng.integers(0, dim)
        cross_mask[j_rand] = True

        trial = np.where(cross_mask, mutant, pop[target_idx])

        for c in centers:
            if c < len(trial):
                trial[c] = pop[target_idx][c]

        trial = np.clip(trial, lower, upper)
        trial[-integer_count:] = np.trunc(trial[-integer_count:])
        return trial

    print("\nROVER OPTIMIZATION WITH DIFFERENTIAL EVOLUTION (DE) & PARALLELISM")
    print("=" * 64)
    print(f"Initial Fitness : {initial_fitness:.9f}")
    print(f"Population      : {POP_SIZE} | Generations: {GENERATIONS}")
    print(f"DE Parameters   : CR={CR}, F={F} | CPU Cores: {MAX_WORKERS}")
    print("=" * 64)

    population = [best.copy()]
    for _ in range(POP_SIZE - 1):
        noise = rng.normal(0, 0.005, size=len(best))
        ind = np.clip(best + noise, lower, upper)
        ind[-integer_count:] = np.trunc(ind[-integer_count:])
        for c in centers:
            if c < len(ind):
                ind[c] = best[c]
        population.append(ind)

    interrupted = False
    try:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            pop_fitness = evaluate_batch(population, executor)

            best_idx = int(np.argmin(pop_fitness))
            if pop_fitness[best_idx] < best_fitness - 1e-6:
                best_fitness = pop_fitness[best_idx]
                best = population[best_idx].copy()

            for gen in range(1, GENERATIONS + 1):
                trial_pop = []
                for i in range(POP_SIZE):
                    trial = mutate_de(population, i, best_idx)
                    trial_pop.append(trial)

                trial_fitness = evaluate_batch(trial_pop, executor)

                for i in range(POP_SIZE):
                    if trial_fitness[i] <= pop_fitness[i]:
                        population[i] = trial_pop[i]
                        pop_fitness[i] = trial_fitness[i]

                        if trial_fitness[i] < best_fitness - 1e-6:
                            best_fitness = trial_fitness[i]
                            best = trial_pop[i].copy()
                            improvements += 1

                            tmp = OUTPUT + ".tmp.npy"
                            np.save(tmp, best)
                            os.replace(tmp, OUTPUT)
                            print(f"\n*** NEW BEST FITNESS: {best_fitness:.9f} ***")

                best_idx = int(np.argmin(pop_fitness))
                print(
                    f"Generation {gen:3d}/{GENERATIONS} | "
                    f"Pop Best: {pop_fitness[best_idx]:.9f} | "
                    f"Global Best: {best_fitness:.9f} | "
                    f"Evaluations: {evaluations}"
                )

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Any improved checkpoint is preserved.")

    print("\n" + "=" * 64)
    print(
        "DE OPTIMIZATION INTERRUPTED"
        if interrupted
        else "DE OPTIMIZATION COMPLETED"
    )
    print("=" * 64)
    print(f"Initial Fitness   : {initial_fitness:.9f}")
    print(f"Best Fitness      : {best_fitness:.9f}")
    print(f"Improvements      : {improvements}")
    print(f"Full Evaluations  : {evaluations}")
    print(f"Cache Hits        : {cache_hits}")
    print(f"Total Improvement : {initial_fitness - best_fitness:.9f}")
    if improvements:
        print(f"Saved Output File : {OUTPUT}")
    else:
        print("No improvement. No new file saved.")

if __name__ == "__main__":
    main()
