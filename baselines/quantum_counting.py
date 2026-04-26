"""
quantum_counting.py
===================
Baseline: Quantum Counting Algorithm (Brassard et al. 2002)

Combines Grover's algorithm with Quantum Phase Estimation to
COUNT (not search) the number of solutions k.

In EGAAT, we use a lightweight MLAE instead of full quantum counting
because quantum counting requires deep circuits (QFT overhead).
This module implements the full version for comparison.

Reference: Brassard et al., Contemporary Mathematics 305, 2002
           EGAAT §2.2
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from typing import List, Tuple


def quantum_counting(
    oracle: 'QuantumCircuit',
    n_qubits: int,
    n_counting: int = 5,
    shots: int = 4096,
    backend=None
) -> Tuple[int, float]:
    """
    Quantum Counting: estimate k using QPE on the Grover operator Q.

    The Grover operator Q has eigenvalues e^(±2iθ) where sin²(θ) = k/N.
    QPE extracts θ → k = N·sin²(θ).

    Parameters
    ----------
    oracle : QuantumCircuit
        Phase oracle for the search problem.
    n_qubits : int
        Number of search qubits.
    n_counting : int
        Number of counting (ancilla) qubits. Precision ∝ 1/2^n_counting.
    shots : int
    backend : optional

    Returns
    -------
    Tuple[int, float]
        (k_estimated, theta_estimated)
    """
    if backend is None:
        backend = AerSimulator(method='statevector')

    N = 2 ** n_qubits

    count_reg = QuantumRegister(n_counting, 'count')
    search_reg = QuantumRegister(n_qubits, 'search')
    creg = ClassicalRegister(n_counting, 'meas')

    qc = QuantumCircuit(count_reg, search_reg, creg, name='QuantumCounting')

    # Initialise search register in uniform superposition |s>
    qc.h(search_reg)

    # Hadamard on counting register
    qc.h(count_reg)

    # Controlled-Q^(2^j) for each counting qubit j
    # Q = D · O (Grover operator)
    grover_op = _build_grover_op(oracle, n_qubits)
    for j in range(n_counting):
        power = 2 ** j
        cQ = grover_op.power(power).control(1)
        qc.append(cQ, [count_reg[j]] + list(search_reg))

    # Inverse QFT on counting register
    iqft = QFT(n_counting, inverse=True)
    qc.append(iqft, count_reg)

    # Measure
    qc.measure(count_reg, creg)

    # Execute
    transpiled = transpile(qc, backend, optimization_level=1)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # Decode
    best_bitstring = max(counts, key=counts.get)
    phase_int = int(best_bitstring, 2)
    M = 2 ** n_counting
    theta = phase_int * np.pi / M
    k_est = N * np.sin(theta) ** 2

    return int(round(np.clip(k_est, 0, N))), theta


def _build_grover_op(oracle, n_qubits):
    """Build Grover operator Q = D · O as a single circuit."""
    from src.diffusion.standard_grover_diffusion import grover_diffusion
    qc = QuantumCircuit(n_qubits, name='Q')
    qc.compose(oracle, inplace=True)
    qc.compose(grover_diffusion(n_qubits), inplace=True)
    return qc


def counting_overhead_vs_mlae(n_qubits: int) -> dict:
    """
    Compare circuit depth overhead of quantum counting vs MLAE.
    Illustrates why EGAAT prefers MLAE for NISQ hardware.
    """
    N = 2 ** n_qubits
    n_counting = 6

    # Quantum counting: QPE needs controlled-Q^(2^j) for j=0..n_counting-1
    # Each Q has depth ≈ n*8; controlled version adds overhead
    qpe_depth = sum(2**j * n_qubits * 8 for j in range(n_counting))
    qft_depth = n_counting ** 2  # QFT depth

    # MLAE: just runs forward Grover circuits (no QPE, no QFT)
    mlae_depth_per_point = n_qubits * 8  # one Grover iteration
    mlae_schedule_points = 7  # typical exponential schedule length
    mlae_total = mlae_depth_per_point * mlae_schedule_points

    return {
        'N': N,
        'n_qubits': n_qubits,
        'quantum_counting_depth': qpe_depth + qft_depth,
        'mlae_total_depth': mlae_total,
        'depth_reduction_factor': (qpe_depth + qft_depth) / mlae_total
    }


if __name__ == "__main__":
    print("=== Quantum Counting vs MLAE Overhead ===\n")
    for n in [4, 6, 8, 10]:
        d = counting_overhead_vs_mlae(n)
        print(f"n={n}, N={d['N']}: "
              f"QCounting depth={d['quantum_counting_depth']}, "
              f"MLAE depth={d['mlae_total_depth']}, "
              f"reduction={d['depth_reduction_factor']:.1f}x")
