"""
circuit_depth.py
================
Experiment: Circuit depth comparison across all algorithms.

EGAAT vs Standard Grover vs Fixed-Point Grover.
Key claim: EGAAT achieves similar or better success probability
with shallower circuits than Fixed-Point Grover.

Reproduces Figure 3 from the EGAAT paper.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import numpy as np
import json
import random
from src.egaat.egaat_circuit import EGAATCircuit
from baselines.standard_grover import standard_grover_search
from baselines.fixed_point_grover import fixed_point_grover_search


def measure_depths(n_qubits_range=None, k=4, trials=3, shots=1024, seed=42):
    """
    Measure actual Qiskit circuit depths for each algorithm.
    Returns dict keyed by algorithm with lists of (N, depth) tuples.
    """
    if n_qubits_range is None:
        n_qubits_range = [4, 5, 6, 7, 8]

    random.seed(seed)
    np.random.seed(seed)

    data = {alg: [] for alg in ['egaat', 'grover', 'fixed_point']}

    for n in n_qubits_range:
        N = 2 ** n
        actual_k = min(k, N // 4)
        targets = random.sample(range(N), actual_k)

        # EGAAT depth (from circuit object)
        egaat = EGAATCircuit(n, targets, shots=shots)
        r = egaat.run()
        data['egaat'].append({'N': N, 'depth': r.circuit_depth, 'p': r.success_probability_measured})

        # Grover depth
        _, p_g, d_g = standard_grover_search(n, targets, shots=shots)
        data['grover'].append({'N': N, 'depth': d_g, 'p': p_g})

        # Fixed-Point depth
        _, p_fp, d_fp = fixed_point_grover_search(n, targets, shots=shots)
        data['fixed_point'].append({'N': N, 'depth': d_fp, 'p': p_fp})

        print(f"N={N:5d}: EGAAT depth={r.circuit_depth:4d}(p={r.success_probability_measured:.3f}) "
              f"Grover={d_g:4d}(p={p_g:.3f}) FP={d_fp:4d}(p={p_fp:.3f})")

    out = 'results/tables/circuit_depth_comparison.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {out}")
    return data


def depth_efficiency_score(data: dict) -> dict:
    """
    Compute depth-efficiency: success_probability / circuit_depth.
    Higher is better — more success per gate.
    """
    scores = {}
    for alg, records in data.items():
        efficiencies = [r['p'] / max(r['depth'], 1) for r in records]
        scores[alg] = {
            'mean_efficiency': float(np.mean(efficiencies)),
            'values': efficiencies
        }
    return scores


if __name__ == "__main__":
    print("=== Circuit Depth Comparison ===\n")
    data = measure_depths(n_qubits_range=[4, 5, 6, 7], k=4)
    scores = depth_efficiency_score(data)
    print("\nDepth Efficiency (P/depth, higher = better):")
    for alg, s in scores.items():
        print(f"  {alg}: {s['mean_efficiency']:.6f}")
