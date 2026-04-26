"""
standard_grover.py
==================
Baseline: Standard Grover's Algorithm (Grover 1996)

Single-target search. Fixed iteration count m = floor(π/4 · sqrt(N)).
This is the comparison baseline for EGAAT in the paper.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from typing import List, Tuple


def standard_grover_search(
    n_qubits: int,
    targets: List[int],
    shots: int = 2048,
    backend=None
) -> Tuple[List[int], float, int]:
    """
    Standard Grover search with fixed iteration count.

    Does NOT adapt to multi-target or unknown k — this is the
    fundamental limitation that EGAAT addresses.

    Returns
    -------
    Tuple[List[int], float, int]
        (top_results, success_probability, circuit_depth)
    """
    if backend is None:
        backend = AerSimulator(method='statevector')

    N = 2 ** n_qubits
    # Fixed m — assumes single target (k=1)
    m = int(np.floor(np.pi / 4 * np.sqrt(N)))

    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr, name='StandardGrover')

    # Superposition
    qc.h(qr)

    # Grover iterations
    for _ in range(m):
        _apply_oracle(qc, qr, targets, n_qubits)
        _apply_diffusion(qc, qr, n_qubits)

    qc.measure(qr, cr)

    transpiled = transpile(qc, backend, optimization_level=1)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # Success probability
    target_shots = sum(
        counts.get(format(t, f'0{n_qubits}b'), 0) for t in targets
    )
    p_success = target_shots / shots

    top = sorted(counts, key=counts.get, reverse=True)[:5]
    found = [int(b, 2) for b in top]

    return found, p_success, qc.depth()


def _apply_oracle(qc, qr, targets, n):
    """Explicit phase oracle for standard Grover."""
    for t in targets:
        bits = format(t, f'0{n}b')
        for i, b in enumerate(reversed(bits)):
            if b == '0':
                qc.x(qr[i])
        qc.h(qr[n - 1])
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(qr[n - 1])
        for i, b in enumerate(reversed(bits)):
            if b == '0':
                qc.x(qr[i])


def _apply_diffusion(qc, qr, n):
    """Standard Grover diffusion."""
    qc.h(qr)
    qc.x(qr)
    qc.h(qr[n - 1])
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(qr[n - 1])
    qc.x(qr)
    qc.h(qr)


if __name__ == "__main__":
    n, targets = 4, [5, 11]
    found, p, depth = standard_grover_search(n, targets)
    print(f"Standard Grover | n={n}, targets={targets}")
    print(f"Found: {found[:3]}, P(success)={p:.4f}, depth={depth}")
    print(f"NOTE: Fixed m={int(np.floor(np.pi/4*np.sqrt(2**n)))} assumes k=1 — degrades for k>1")
