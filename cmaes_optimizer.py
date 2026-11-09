import os
import cma
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from morphing_udp import (
    udp,
    NUM_MODE_PARAMETERS,
    NUM_BIASES,
    NUM_WEIGHTS,
    MASK_CENTRES,
)

# Konfigurimi i skedarëve
PRIMARY_SOURCE = "final_results/adaptive_hill_best_rover.npy"
FALLBACK_SOURCES = [
    "final_results/multistart_ga_v4_best_rover.npy",
]
OUTPUT = "final_results/cmaes_deep_best_rover.npy"

RANDOM_SEED = 2048
MAX_WORKERS = os.cpu_count() or 8

# Hiperparametrat për IPOP-CMA-ES
INITIAL_SIGMA = 0.05       # Hapi fillestar i mjaftueshëm për të dalë nga 3.6x
BASE_POP_SIZE = 40        # Madhësia bazë e popullatës
MAX_ITER_PER_RUN = 250    # Iteracionet për çdo fazë ri-nisjeje
MAX_RESTARTS = 4          # Numri i ri-nisjeve me rritje popullate (IPOP)

# Variablat globale për worker-at parallel
_global_base_rover = None
_global_active_indices = None
_global_lower_bound = None
_global_upper_bound = None


def _init_worker(base_rover, active_indices, lower_b, upper_b):
    """Inicializon çdo worker me template-in bazë për të shmangur overhead-in IPC."""
    global _global_base_rover, _global_active_indices, _global_lower_bound, _global_upper_bound
    _global_base_rover = base_rover.copy()
    _global_active_indices = active_indices
    _global_lower_bound = lower_b
    _global_upper_bound = upper_b


def _evaluate_active_candidate(candidate_active):
    """
    Krasit vektorin aktiv me kufijtë dhe e vendos te rover-i bazë.
    Kjo nuk e dëmton meksnizmin e brendshëm të CMA-ES.
    """
    cand_clipped = np.clip(candidate_active, _global_lower_bound, _global_upper_bound)
    
    full_candidate = _global_base_rover.copy()
    full_candidate[_global_active_indices] = cand_clipped
    
    fit = udp.fitness(full_candidate)[0]
    return float(fit) if np.isfinite(fit) else 1e9


def resolve_source_file():
    if os.path.isfile(PRIMARY_SOURCE):
        return PRIMARY_SOURCE
    for src in FALLBACK_SOURCES:
        if os.path.isfile(src):
            return src
    raise SystemExit(f"Nuk u gjet asnjë skedar burim. Kontrolluar: {PRIMARY_SOURCE}")


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    source_file = resolve_source_file()

    # Marrja e kufijve të plotë
    lower_full, upper_full = udp.get_bounds()
    lower_full = np.asarray(lower_full, dtype=float)
    upper_full = np.asarray(upper_full, dtype=float)

    best_full = np.load(source_file, allow_pickle=False).copy()
    initial_fitness = float(udp.fitness(best_full)[0])

    if not np.isfinite(initial_fitness):
        raise SystemExit("Fitnesi fillestar nuk është numër i fundmë.")

    # Izolimi i indekseve aktive (përjashtohen MASK_CENTRES)
    mask_set = set(MASK_CENTRES)
    active_indices = np.array([i for i in range(len(best_full)) if i not in mask_set])
    
    # Kufijtë dhe pika fillestare vetëm për parametrat aktivë
    lower_active = lower_full[active_indices]
    upper_active = upper_full[active_indices]
    x0_active = best_full[active_indices]

    global_best_fitness = initial_fitness
    global_best_full = best_full.copy()
    total_improvements = 0

    print("\n" + "=" * 72)
    print("IPOP-CMA-ES DEEP OPTIMIZER (TARGETING 2.x / 1.x FITNESS)")
    print("=" * 72)
    print(f"Skedari Burim      : {source_file}")
    print(f"Fitnesi Fillestar  : {initial_fitness:.9f}")
    print(f"Dimensioni i plotë : {len(best_full)} | Dimensioni Aktiv: {len(active_indices)}")
    print(f"Cores CPU          : {MAX_WORKERS} | Ri-nisje Max (IPOP): {MAX_RESTARTS}")
    print("=" * 72)

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(best_full, active_indices, lower_active, upper_active),
    ) as executor:

        popsize = BASE_POP_SIZE
        
        for restart_idx in range(MAX_RESTARTS + 1):
            sigma = INITIAL_SIGMA * (1.2 ** restart_idx)
            
            opts = {
                "bounds": [lower_active, upper_active],
                "seed": RANDOM_SEED + restart_idx * 100,
                "popsize": popsize,
                "maxiter": MAX_ITER_PER_RUN,
                "verbose": -1,
                "tolfun": 1e-10,
                "tolx": 1e-10,
            }

            # Në ri-nisjet e mëvonshme nisemi nga zgjidhja më e mirë e arritur deri tani
            start_x = global_best_full[active_indices]
            es = cma.CMAEvolutionStrategy(start_x, sigma, opts)
            iteration = 0

            print(f"\n--- Fillon Faza/Ri-nisja #{restart_idx + 1} (Popsize: {popsize}, Sigma: {sigma:.4f}) ---")

            while not es.stop():
                iteration += 1
                candidates = es.ask()

                # Vlerësimi parallel i kandidatëve pa ndryshuar vektorët e pyetur nga ask()
                fitnesses = list(
                    executor.map(_evaluate_active_candidate, candidates, chunksize=1)
                )

                # Përditësimi i saktë i CMA-ES me kandidatët origjinalë
                es.tell(candidates, fitnesses)

                min_gen_fit = min(fitnesses)
                min_gen_idx = fitnesses.index(min_gen_fit)

                if min_gen_fit < global_best_fitness - 1e-6:
                    global_best_fitness = min_gen_fit
                    
                    # Rikonstruktimi i vektorit të plotë për ta ruajtur
                    best_cand_active = np.clip(candidates[min_gen_idx], lower_active, upper_active)
                    global_best_full[active_indices] = best_cand_active
                    total_improvements += 1

                    # Ruajtja atomike në disk
                    tmp = OUTPUT + ".tmp.npy"
                    np.save(tmp, global_best_full)
                    os.replace(tmp, OUTPUT)

                    print(
                        f"*** REKORD I RI FITNESI: {global_best_fitness:.9f} (Ri-nisja {restart_idx + 1}, Iter {iteration}) ***"
                    )

                if iteration % 10 == 0 or min_gen_fit == global_best_fitness:
                    print(
                        f"Ri-nisja {restart_idx + 1} | Iter {iteration:3d}/{MAX_ITER_PER_RUN} | "
                        f"Gen Best: {min_gen_fit:.7f} | Global Best: {global_best_fitness:.9f} | "
                        f"Sigma: {es.sigma:.6f}"
                    )

            # Rritja e popullatës për ri-nisjen tjetër (IPOP Strategy)
            popsize = int(popsize * 1.5)

    print("\n" + "=" * 72)
    print("PËRMBLEDHJA E OPTIMIZIMIT IPOP-CMA-ES")
    print("=" * 72)
    print(f"Fitnesi Fillestar   : {initial_fitness:.9f}")
    print(f"Fitnesi Më i Mirë   : {global_best_fitness:.9f}")
    print(f"Përmirësimi Total   : {initial_fitness - global_best_fitness:.9f}")
    print(f"Përmirësime me Sukses: {total_improvements}")
    if total_improvements > 0:
        print(f"Skedari i Ruajtur   : {OUTPUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()