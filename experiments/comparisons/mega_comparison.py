"""
mega_comparison.py
==================
Comparison of Classical Search vs Standard Grover vs EGAAT.
Shows why EGAAT is necessary for multi-target databases.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from src.egaat.egaat_circuit import EGAATCircuit
from src.iteration_control.dynamic_iteration import optimal_iterations, success_probability

def run_comparison():
    N = 128
    n_qubits = 7
    k_range = [1, 2, 4, 8, 16, 32]
    
    results = []

    print(f"{'k':>3} | {'Classical':>10} | {'Std Grover P':>12} | {'EGAAT P':>10} | {'EGAAT m*':>8}")
    print("-" * 60)

    for k in k_range:
        # 1. Classical Query Complexity (avg)
        classical_queries = N / (k + 1) # Expected search time

        # 2. Standard Grover Success (assuming they fixed m for k=1)
        m_fixed = int(np.floor(np.pi/4 * np.sqrt(N/1))) # Naive fixed m
        p_std = success_probability(N, k, m_fixed)

        # 3. EGAAT Performance
        targets = list(range(k))
        egaat = EGAATCircuit(n_qubits=n_qubits, targets=targets, shots=1024)
        res = egaat.run()
        
        results.append({
            'k': k,
            'classical_q': classical_queries,
            'std_p': p_std,
            'egaat_p': res.success_probability_measured,
            'egaat_m': res.iterations_used
        })

        print(f"{k:3d} | {classical_queries:10.1f} | {p_std:12.4f} | {res.success_probability_measured:10.4f} | {res.iterations_used:8d}")

    # Plotting
    plt.figure(figsize=(10, 6))
    ks = [r['k'] for r in results]
    plt.plot(ks, [r['std_p'] for r in results], 'ro--', label='Standard Grover (Fixed m)')
    plt.plot(ks, [r['egaat_p'] for r in results], 'gs-', label='EGAAT (Adaptive)')
    plt.axhline(y=0.94, color='blue', linestyle=':', label='94% Threshold')
    
    plt.title(f"Success Probability vs Target Count (N={N})")
    plt.xlabel("Number of Targets (k)")
    plt.ylabel("Success Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/egaat_vs_std_comparison.png')
    print(f"\nPlot saved to results/figures/egaat_vs_std_comparison.png")

if __name__ == "__main__":
    run_comparison()
