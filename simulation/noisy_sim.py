"""
noisy_sim.py
============
Noisy Qiskit simulation using Aer with depolarising noise.
Used for NISQ validation experiments in EGAAT §1.5, §2.5.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from experiments.noise_models.depolarising_noise import build_depolarising_model


def get_noisy_backend(p1: float = 0.001, p2: float = None) -> AerSimulator:
    """
    Return an AerSimulator with depolarising noise.

    Parameters
    ----------
    p1 : float
        Single-qubit error rate.
    p2 : float, optional
        Two-qubit error rate. Defaults to 10×p1.
    """
    if p2 is None:
        p2 = p1 * 10
    nm = build_depolarising_model(p1=p1, p2=p2)
    return AerSimulator(noise_model=nm)


def run_noisy_circuit(qc, p1: float = 0.001, shots: int = 2048) -> dict:
    """Run a circuit with noise and return measurement counts."""
    backend = get_noisy_backend(p1)
    transpiled = transpile(qc, backend, optimization_level=1)
    result = backend.run(transpiled, shots=shots).result()
    return result.get_counts()


if __name__ == "__main__":
    from src.egaat.egaat_circuit import EGAATCircuit
    print("=== Noisy Simulation Test ===\n")

    for p1 in [0.0, 0.0005, 0.001, 0.005]:
        backend = get_noisy_backend(p1)
        egaat = EGAATCircuit(4, [3, 9], backend=backend, shots=2048)
        r = egaat.run()
        print(f"p1={p1:.4f}: P(success)={r.success_probability_measured:.4f}")
