"""
generate_metrics.py
===================
End-to-end performance benchmarking for EGAAT.
Tests various database sizes (N) and target densities (k/N).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.egaat.egaat_circuit import EGAATCircuit

def run_benchmark():
    test_cases = [
        # (n_qubits, num_targets)
        (4, 1),   # N=16,  k=1 (Sparse)
        (4, 4),   # N=16,  k=4 (Dense)
        (5, 2),   # N=32,  k=2 
        (5, 6),   # N=32,  k=6
        (6, 1),   # N=64,  k=1
        (6, 8),   # N=64,  k=8 (k = sqrt(N))
        (7, 4),   # N=128, k=4
        (7, 16),  # N=128, k=16 (Large)
    ]

    results = []

    print(f"{'n':>2} {'N':>4} {'True k':>6} {'Est k':>6} {'m*':>4} {'P(success)':>10} {'Depth':>6}")
    print("-" * 55)

    for n, k_count in test_cases:
        # Generate random targets
        targets = list(np.random.choice(range(2**n), k_count, replace=False))
        targets = [int(t) for t in targets]

        # Run EGAAT
        egaat = EGAATCircuit(n_qubits=n, targets=targets, shots=2048, estimator_method='qae')
        res = egaat.run()

        results.append({
            'n_qubits': n,
            'N': 2**n,
            'k_true': len(targets),
            'k_est': res.k_estimated,
            'iterations': res.iterations_used,
            'p_success': res.success_probability_measured,
            'depth': res.circuit_depth
        })

        print(f"{n:2d} {2**n:4d} {len(targets):6d} {res.k_estimated:6d} {res.iterations_used:4d} "
              f"{res.success_probability_measured:10.4f} {res.circuit_depth:6d}")

    # Save to CSV for the user
    df = pd.DataFrame(results)
    df.to_csv('results/performance_metrics.csv', index=False)
    print(f"\nMetrics saved to results/performance_metrics.csv")

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    run_benchmark()
