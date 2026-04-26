"""
db_4096.py — N=4096 (n=12 qubits) benchmark
Upper bound from EGAAT abstract: databases up to 4096 entries.
Uses fewer trials due to simulation cost at this scale.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import numpy as np, random, json
from src.egaat.egaat_circuit import EGAATCircuit
from src.iteration_control.dynamic_iteration import optimal_iterations, success_probability

N_QUBITS, N, SHOTS, TRIALS, SEED = 12, 4096, 512, 2, 42
TARGET_COUNTS = [4, 16, 64, 256, 512, 1024]

def run():
    random.seed(SEED); np.random.seed(SEED)
    results = []
    for k in TARGET_COUNTS:
        ep = []
        for _ in range(TRIALS):
            targets = random.sample(range(N), k)
            r = EGAATCircuit(N_QUBITS, targets, shots=SHOTS).run()
            ep.append(r.success_probability_measured)

        # Also compute theoretical for comparison
        m = optimal_iterations(N, k)
        p_theory = success_probability(N, k, m)

        row = dict(N=N, k=k, density=round(k/N, 4),
                   egaat=round(float(np.mean(ep)), 4),
                   theoretical=round(p_theory, 4),
                   optimal_m=m)
        results.append(row)
        print(f"N=4096, k={k:5d}: EGAAT={row['egaat']} Theory={row['theoretical']} m*={m}")

    path = 'results/tables/db4096_results.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(results, open(path,'w'), indent=2)
    print(f"Saved: {path}")
    return results

if __name__ == "__main__":
    print("=== N=4096 Benchmark ===\n"); run()
