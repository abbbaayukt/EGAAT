"""
db_256.py — N=256 (n=8 qubits) benchmark
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import numpy as np, random, json
from src.egaat.egaat_circuit import EGAATCircuit
from baselines.standard_grover import standard_grover_search
from baselines.fixed_point_grover import fixed_point_grover_search

N_QUBITS, N, SHOTS, TRIALS, SEED = 8, 256, 2048, 5, 42
TARGET_COUNTS = [1, 4, 8, 16, 32, 64, 128]

def run():
    random.seed(SEED); np.random.seed(SEED)
    results = []
    for k in TARGET_COUNTS:
        ep, gp, fp = [], [], []
        for _ in range(TRIALS):
            targets = random.sample(range(N), k)
            r = EGAATCircuit(N_QUBITS, targets, shots=SHOTS).run()
            ep.append(r.success_probability_measured)
            _, p, _ = standard_grover_search(N_QUBITS, targets, shots=SHOTS)
            gp.append(p)
            _, p, _ = fixed_point_grover_search(N_QUBITS, targets, shots=SHOTS)
            fp.append(p)
        row = dict(N=N, k=k, density=k/N,
                   egaat=round(float(np.mean(ep)),4),
                   grover=round(float(np.mean(gp)),4),
                   fp=round(float(np.mean(fp)),4))
        results.append(row)
        print(f"N=256, k={k:3d}: EGAAT={row['egaat']} Grover={row['grover']} FP={row['fp']}")
    path = 'results/tables/db256_results.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(results, open(path,'w'), indent=2)
    print(f"Saved: {path}")
    return results

if __name__ == "__main__":
    print("=== N=256 Benchmark ===\n"); run()
