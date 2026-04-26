"""
db_1024.py — N=1024 (n=10 qubits) benchmark
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import numpy as np, random, json
from src.egaat.egaat_circuit import EGAATCircuit
from baselines.standard_grover import standard_grover_search

N_QUBITS, N, SHOTS, TRIALS, SEED = 10, 1024, 1024, 3, 42
TARGET_COUNTS = [1, 4, 16, 64, 128, 256, 512]

def run():
    random.seed(SEED); np.random.seed(SEED)
    results = []
    for k in TARGET_COUNTS:
        ep, gp = [], []
        for _ in range(TRIALS):
            targets = random.sample(range(N), k)
            r = EGAATCircuit(N_QUBITS, targets, shots=SHOTS).run()
            ep.append(r.success_probability_measured)
            _, p, _ = standard_grover_search(N_QUBITS, targets, shots=SHOTS)
            gp.append(p)
        row = dict(N=N, k=k, density=k/N,
                   egaat=round(float(np.mean(ep)),4),
                   grover=round(float(np.mean(gp)),4),
                   circuit_depth_mean=r.circuit_depth)
        results.append(row)
        print(f"N=1024, k={k:4d}: EGAAT={row['egaat']} Grover={row['grover']}")
    path = 'results/tables/db1024_results.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(results, open(path,'w'), indent=2)
    print(f"Saved: {path}")
    return results

if __name__ == "__main__":
    print("=== N=1024 Benchmark ===\n"); run()
