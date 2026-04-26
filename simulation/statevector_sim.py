"""
statevector_sim.py
==================
Statevector-level analysis tools for EGAAT.

Allows exact inspection of quantum state amplitudes at each
Grover iteration — useful for debugging and visualising amplitude
amplification progress.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from typing import List


def get_amplitude_evolution(
    oracle: QuantumCircuit,
    diffusion: QuantumCircuit,
    n_qubits: int,
    targets: List[int],
    max_iter: int = 20
) -> dict:
    """
    Track amplitude of target states across Grover iterations.

    Returns dict with iteration counts and corresponding target amplitudes.
    Used to verify adaptive diffusion behaviour.
    """
    N = 2 ** n_qubits
    amplitudes = []
    non_target_amps = []

    qc_base = QuantumCircuit(n_qubits)
    qc_base.h(range(n_qubits))

    sv = Statevector.from_instruction(qc_base)

    for i in range(max_iter + 1):
        probs = np.abs(sv.data) ** 2
        target_prob = sum(probs[t] for t in targets)
        non_target_prob = 1.0 - target_prob

        amplitudes.append(float(target_prob))
        non_target_amps.append(float(non_target_prob))

        if i < max_iter:
            # Apply one Grover iteration
            iteration = QuantumCircuit(n_qubits)
            iteration.compose(oracle, inplace=True)
            iteration.compose(diffusion, inplace=True)
            sv = sv.evolve(iteration)

    return {
        'iterations': list(range(max_iter + 1)),
        'target_probability': amplitudes,
        'non_target_probability': non_target_amps,
        'peak_iteration': int(np.argmax(amplitudes)),
        'peak_probability': float(np.max(amplitudes))
    }


def compare_amplitude_evolution(n_qubits: int, targets: List[int]) -> dict:
    """
    Compare amplitude evolution: standard diffusion vs adaptive diffusion.
    Shows why EGAAT's α-scheduling outperforms fixed diffusion.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.oracle.multi_target_oracle import build_multi_target_oracle
    from src.diffusion.adaptive_diffusion import grover_diffusion, adaptive_diffusion, compute_alpha

    oracle = build_multi_target_oracle(n_qubits, targets)
    N = 2 ** n_qubits
    k = len(targets)

    # Standard diffusion (α=1.0)
    std_diff = grover_diffusion(n_qubits)
    std_evo = get_amplitude_evolution(oracle, std_diff, n_qubits, targets)

    # Adaptive diffusion
    alpha = compute_alpha(k, N)
    ada_diff = adaptive_diffusion(n_qubits, alpha)
    ada_evo = get_amplitude_evolution(oracle, ada_diff, n_qubits, targets)

    return {
        'standard': std_evo,
        'adaptive': ada_evo,
        'alpha_used': alpha,
        'improvement': ada_evo['peak_probability'] - std_evo['peak_probability']
    }


if __name__ == "__main__":
    n, targets = 4, [3, 9, 14]
    print(f"=== Amplitude Evolution: n={n}, targets={targets} ===\n")

    evo = compare_amplitude_evolution(n, targets)
    print(f"Standard diffusion peak: {evo['standard']['peak_probability']:.4f} "
          f"at iter {evo['standard']['peak_iteration']}")
    print(f"Adaptive diffusion peak: {evo['adaptive']['peak_probability']:.4f} "
          f"at iter {evo['adaptive']['peak_iteration']} (α={evo['alpha_used']:.2f})")
    print(f"Improvement: {evo['improvement']:+.4f}")
