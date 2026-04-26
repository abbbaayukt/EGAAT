"""
depolarising_noise.py
=====================
NISQ noise model: depolarising noise on single and two-qubit gates.

Used to validate EGAAT robustness under realistic NISQ hardware conditions
as described in EGAAT §1.5 contribution 5 and §2.5.

Finding from paper: EGAAT stable for noise rates up to 0.1%.
"""

import numpy as np
from typing import Optional
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, thermal_relaxation_error
)
from qiskit_aer import AerSimulator


def build_depolarising_model(
    p1: float = 0.001,
    p2: float = 0.01
) -> NoiseModel:
    """
    Build a depolarising noise model.

    Parameters
    ----------
    p1 : float
        Single-qubit gate error rate. Default 0.1% (1e-3).
    p2 : float
        Two-qubit gate error rate. Default 1% (1e-2).
        Standard relationship: p2 ≈ 10×p1 for NISQ hardware.

    Returns
    -------
    NoiseModel
    """
    noise_model = NoiseModel()

    # Single-qubit gate errors
    error_1q = depolarizing_error(p1, 1)
    for gate in ['h', 'x', 'z', 'rz', 'rx', 'ry', 'u']:
        noise_model.add_all_qubit_quantum_error(error_1q, gate)

    # Two-qubit gate errors
    error_2q = depolarizing_error(p2, 2)
    for gate in ['cx', 'cz']:
        noise_model.add_all_qubit_quantum_error(error_2q, gate)

    # Three-qubit gate errors (e.g., CCX)
    error_3q = depolarizing_error(p2 * 2, 3)  # CCX is typically more noisy
    noise_model.add_all_qubit_quantum_error(error_3q, 'ccx')

    return noise_model


def build_thermal_model(
    t1: float = 50e-6,
    t2: float = 70e-6,
    gate_time_1q: float = 50e-9,
    gate_time_2q: float = 300e-9
) -> NoiseModel:
    """
    Build a thermal relaxation noise model (T1/T2 decoherence).

    Parameters
    ----------
    t1 : float
        T1 relaxation time in seconds. Default 50µs.
    t2 : float
        T2 dephasing time in seconds. Default 70µs.
    gate_time_1q : float
        Single-qubit gate time. Default 50ns.
    gate_time_2q : float
        Two-qubit gate time. Default 300ns.

    Returns
    -------
    NoiseModel
    """
    noise_model = NoiseModel()

    error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).expand(
        thermal_relaxation_error(t1, t2, gate_time_2q)
    )

    for gate in ['h', 'x', 'z', 'rz']:
        noise_model.add_all_qubit_quantum_error(error_1q, gate)

    for gate in ['cx', 'cz']:
        noise_model.add_all_qubit_quantum_error(error_2q, gate)

    return noise_model


def noise_sweep(
    n_qubits: int,
    targets: list,
    error_rates: list,
    shots: int = 2048
) -> dict:
    """
    Sweep over noise levels and measure EGAAT success probability.
    Reproduces Fig 4 (noise resilience) from the paper.

    Parameters
    ----------
    n_qubits : int
    targets : list
    error_rates : list of float
        Single-qubit error rates to test.
    shots : int

    Returns
    -------
    dict with keys 'error_rates', 'egaat_probs', 'grover_probs'
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.egaat.egaat_circuit import EGAATCircuit
    from baselines.standard_grover import standard_grover_search

    egaat_probs, grover_probs = [], []

    for p1 in error_rates:
        nm = build_depolarising_model(p1=p1, p2=p1 * 10)
        backend = AerSimulator(noise_model=nm)

        # EGAAT
        egaat = EGAATCircuit(n_qubits, targets, backend=backend, shots=shots)
        r = egaat.run()
        egaat_probs.append(r.success_probability_measured)

        # Standard Grover (baseline)
        _, p_g, _ = standard_grover_search(n_qubits, targets, shots=shots, backend=backend)
        grover_probs.append(p_g)

        print(f"  p1={p1:.4f}: EGAAT={egaat_probs[-1]:.4f}, Grover={grover_probs[-1]:.4f}")

    return {
        'error_rates': error_rates,
        'egaat_probs': egaat_probs,
        'grover_probs': grover_probs
    }


if __name__ == "__main__":
    print("=== NISQ Noise Model Test ===\n")
    nm = build_depolarising_model(p1=0.001, p2=0.01)
    print("Depolarising noise model built:")
    print(f"  Single-qubit gates: {nm.noise_qubits}")
    print(f"  Basis gates: {nm.basis_gates}")

    print("\nRunning noise sweep (p1 = 0% to 0.5%)...")
    error_rates = [0.0, 0.0005, 0.001, 0.002, 0.005]
    results = noise_sweep(4, [3, 9, 14], error_rates, shots=1024)

    print("\nResults:")
    for p, pe, pg in zip(results['error_rates'], results['egaat_probs'], results['grover_probs']):
        print(f"  p1={p:.4f}: EGAAT={pe:.4f}, Grover={pg:.4f}")
