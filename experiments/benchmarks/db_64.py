"""
db_64.py
========
Benchmark: N=64 database (n=6 qubits), all target densities.

Runs EGAAT and all baselines across k=1 to k=32 targets.
Produces per-configuration success probability and circuit depth data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import random
import json
from src.egaat.egaat_circuit import EGAATCircuit
from baselines.standard_grover import standard_grover_search
from baselines.fixed_point_grover import fixed_point_grover_search

N_QUBITS = 6
N = 64
SHOTS = 2048
TRIALS = 5
SEED = 42

TARGET_COUNTS = [1, 2, 4, 8, 16, 32]


def run_db64_benchmark():
    random.seed(SEED)
    np.random.seed(SEED)
    results = []

    for k in TARGET_COUNTS:
        row = {'n_qubits': N_QUBITS, 'N': N, 'k': k, 'density': k/N}
        egaat_p, grover_p, fp_p = [], [], []

        for trial in range(TRIALS):
            targets = random.sample(range(N), k)

            egaat = EGAATCircuit(N_QUBITS, targets, shots=SHOTS)
            r = egaat.run()
            egaat_p.append(r.success_probability_measured)

            _, p_g, _ = standard_grover_search(N_QUBITS, targets, shots=SHOTS)
            grover_p.append(p_g)

            _, p_fp, _ = fixed_point_grover_search(N_QUBITS, targets, shots=SHOTS)
            fp_p.append(p_fp)

        row['egaat_mean'] = float(np.mean(egaat_p))
        row['egaat_std']  = float(np.std(egaat_p))
        row['grover_mean'] = float(np.mean(grover_p))
        row['fp_mean']    = float(np.mean(fp_p))

        results.append(row)
        print(f"N=64, k={k:2d}: EGAAT={row['egaat_mean']:.4f} "
              f"Grover={row['grover_mean']:.4f} FP={row['fp_mean']:.4f}")

    out = 'results/tables/db64_results.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


if __name__ == "__main__":
    print("=== N=64 Benchmark ===\n")
    run_db64_benchmark()
