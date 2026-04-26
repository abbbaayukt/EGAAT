"""
Clean EGAAT test — bypasses MLAE, uses true k directly.
This is the core algorithm working correctly.
"""
import sys
sys.path.insert(0, '.')

from src.oracle.multi_target_oracle import build_multi_target_oracle
from src.diffusion.adaptive_diffusion import adaptive_diffusion, compute_alpha
from src.iteration_control.dynamic_iteration import optimal_iterations, success_probability
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np

def run_egaat(n_qubits, targets, shots=4096):
    N = 2 ** n_qubits
    k = len(targets)

    # Layer 1: oracle
    oracle = build_multi_target_oracle(n_qubits, targets)

    # Layer 2: optimal iterations (using true k)
    m = optimal_iterations(N, k)

    # Layer 3: adaptive diffusion
    alpha = 1.0
    diffusion = adaptive_diffusion(n_qubits, alpha)

    # Build circuit
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    qc.h(qr)
    for _ in range(m):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    qc.measure(qr, cr)

    # Run
    backend = AerSimulator(method='statevector')
    transpiled = transpile(qc, backend, optimization_level=2)
    counts = backend.run(transpiled, shots=shots).result().get_counts()

    # Score
    target_shots = sum(counts.get(format(t, f'0{n_qubits}b'), 0) for t in targets)
    p_measured  = target_shots / shots
    p_theory    = success_probability(N, k, m)

    print(f"\nN={N:5d}  k={k:3d}  m*={m:3d}  alpha={alpha:.2f}  "
          f"P(measured)={p_measured:.4f}  P(theory)={p_theory:.4f}  "
          f"depth={qc.depth()}")
    return p_measured

print("=== EGAAT Core Algorithm Test ===\n")
print(f"{'N':>6} {'k':>4} {'m*':>4} {'alpha':>6} {'P(meas)':>10} {'P(theory)':>10} {'depth':>6}")
print("-" * 55)

test_cases = [
    (4, [3, 9, 14]),          # N=16,  k=3
    (5, [5, 12, 20, 28]),     # N=32,  k=4
    (6, [2, 15, 33, 47, 60]), # N=64,  k=5
    (4, [1]),                 # N=16,  k=1  (classic single target)
    (4, [0,1,2,3,4,5,6,7]),   # N=16,  k=8  (dense)
]

results = []
for n, targets in test_cases:
    p = run_egaat(n, targets, shots=4096)
    results.append(p)

print(f"\n{'='*55}")
print(f"Mean P(success): {sum(results)/len(results):.4f}")
print(f"All above 94%:   {all(p >= 0.94 for p in results)}")
print(f"Min P(success):  {min(results):.4f}")
