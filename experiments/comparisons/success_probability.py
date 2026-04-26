"""
success_probability.py
======================
Experiment: Compare success probability across all algorithms.

EGAAT vs Standard Grover vs Fixed-Point Grover vs Quantum Walk
across database sizes N=64,256,1024,4096 and target densities.

Reproduces Table 1 and Figure 1 from the EGAAT paper.
Target: EGAAT achieves >94% across all configurations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import json
import random
from typing import List, Dict

from src.egaat.egaat_circuit import EGAATCircuit
from baselines.standard_grover import standard_grover_search
from baselines.fixed_point_grover import fixed_point_grover_search


def run_comparison(
    n_qubits: int,
    k: int,
    n_trials: int = 10,
    shots: int = 2048,
    seed: int = 42
) -> Dict:
    """
    Compare all algorithms on a single (N, k) configuration.

    Runs n_trials with random target sets and averages results.
    """
    random.seed(seed)
    np.random.seed(seed)
    N = 2 ** n_qubits

    egaat_probs, grover_probs, fp_probs = [], [], []
    egaat_depths, grover_depths, fp_depths = [], [], []

    for trial in range(n_trials):
        targets = random.sample(range(N), k)

        # EGAAT
        egaat = EGAATCircuit(n_qubits, targets, shots=shots)
        r = egaat.run()
        egaat_probs.append(r.success_probability_measured)
        egaat_depths.append(r.circuit_depth)

        # Standard Grover
        _, p_g, d_g = standard_grover_search(n_qubits, targets, shots=shots)
        grover_probs.append(p_g)
        grover_depths.append(d_g)

        # Fixed-Point Grover
        _, p_fp, d_fp = fixed_point_grover_search(n_qubits, targets, shots=shots)
        fp_probs.append(p_fp)
        fp_depths.append(d_fp)

    return {
        'n_qubits': n_qubits,
        'N': N,
        'k': k,
        'density': k / N,
        'egaat': {
            'mean_prob': float(np.mean(egaat_probs)),
            'std_prob':  float(np.std(egaat_probs)),
            'mean_depth': float(np.mean(egaat_depths)),
        },
        'standard_grover': {
            'mean_prob': float(np.mean(grover_probs)),
            'std_prob':  float(np.std(grover_probs)),
            'mean_depth': float(np.mean(grover_depths)),
        },
        'fixed_point': {
            'mean_prob': float(np.mean(fp_probs)),
            'std_prob':  float(np.std(fp_probs)),
            'mean_depth': float(np.mean(fp_depths)),
        },
    }


def full_benchmark(output_path: str = 'results/tables/table1_comparison.json'):
    """
    Full benchmark across all database sizes from the paper.
    N ∈ {64, 256, 1024, 4096}, varying target densities.
    """
    configs = [
        # (n_qubits, k) — spans sparse to dense
        (6, 1),   (6, 4),   (6, 16),  (6, 32),
        (8, 1),   (8, 8),   (8, 32),  (8, 128),
        (10, 4),  (10, 32), (10, 128),
        (12, 8),  (12, 64), (12, 512),
    ]

    all_results = []
    for n, k in configs:
        print(f"\nN={2**n}, k={k} (density={k/2**n:.3f})")
        r = run_comparison(n, k, n_trials=5, shots=1024)
        all_results.append(r)
        print(f"  EGAAT:   {r['egaat']['mean_prob']:.4f} ± {r['egaat']['std_prob']:.4f}")
        print(f"  Grover:  {r['standard_grover']['mean_prob']:.4f}")
        print(f"  FP:      {r['fixed_point']['mean_prob']:.4f}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Summary
    egaat_all = [r['egaat']['mean_prob'] for r in all_results]
    print(f"\nEGAAT overall: mean={np.mean(egaat_all):.4f}, min={np.min(egaat_all):.4f}")
    print(f"All > 94%: {all(p >= 0.94 for p in egaat_all)}")

    return all_results


if __name__ == "__main__":
    # Quick smoke test first
    print("=== Success Probability Comparison ===\n")
    r = run_comparison(n_qubits=4, k=2, n_trials=3, shots=1024)
    print(f"N=16, k=2:")
    print(f"  EGAAT  P = {r['egaat']['mean_prob']:.4f}")
    print(f"  Grover P = {r['standard_grover']['mean_prob']:.4f}")
    print(f"  FP     P = {r['fixed_point']['mean_prob']:.4f}")
