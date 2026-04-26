"""
egaat_circuit.py
================
EGAAT — Unified 3-Layer Adaptive Quantum Search Circuit

This is the main circuit builder integrating all three contributions:
  Layer 1: Multi-target adaptive oracle (§1.2, §1.5 contrib 1)
  Layer 2: QAE pre-processing + dynamic iteration control (§1.2, contrib 2)
  Layer 3: Adaptive diffusion scheduling (§1.2, contrib 3)

Usage:
    circuit = EGAATCircuit(n_qubits=6, targets=[5, 12, 47])
    result  = circuit.run()
    print(result.found_targets, result.success_probability)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oracle.multi_target_oracle import build_multi_target_oracle
from iteration_control.qae_estimator import estimate_k
from iteration_control.dynamic_iteration import optimal_iterations, success_probability
from diffusion.adaptive_diffusion import adaptive_diffusion, compute_alpha


@dataclass
class EGAATResult:
    """Results from a single EGAAT search execution."""
    found_targets: List[int]
    true_targets: List[int]
    n_qubits: int
    k_estimated: int
    k_true: int
    iterations_used: int
    alpha_used: float
    success_probability_measured: float
    success_probability_theoretical: float
    circuit_depth: int
    shot_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return any(t in self.true_targets for t in self.found_targets)

    @property
    def precision(self) -> float:
        if not self.found_targets:
            return 0.0
        hits = sum(1 for t in self.found_targets if t in self.true_targets)
        return hits / len(self.found_targets)


class EGAATCircuit:
    """
    Enhanced Grover's Algorithm with Adaptive Techniques.

    Implements the complete EGAAT framework:
    1. Build multi-target oracle
    2. Run MLAE pre-processing to estimate k
    3. Compute optimal iterations using dynamic controller
    4. Apply Grover iterations with adaptive diffusion operator
    5. Measure and return results

    Parameters
    ----------
    n_qubits : int
        log2(N) — number of qubits. Database size N = 2^n_qubits.
    targets : List[int]
        True target indices. In a real application these are unknown;
        they are provided here only for simulation validation.
    estimator_method : str
        'mlae' (default, NISQ) or 'qae' (full precision).
    backend : optional
        Qiskit backend. Defaults to AerSimulator.
    shots : int
        Measurement shots for final search.
    """

    def __init__(
        self,
        n_qubits: int,
        targets: List[int],
        estimator_method: str = 'qae',
        backend=None,
        shots: int = 2048,
        noise_model=None
    ):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.targets = targets
        self.estimator_method = estimator_method
        self.shots = shots
        self.noise_model = noise_model

        if backend is None:
            if noise_model:
                self.backend = AerSimulator(noise_model=noise_model)
            else:
                self.backend = AerSimulator(method='statevector')
        else:
            self.backend = backend

    def build_search_circuit(self, m: int, alpha: float) -> QuantumCircuit:
        """
        Construct the full EGAAT search circuit with m adaptive iterations.

        Circuit structure:
            H^n                    — initial superposition
            [O · D(α)]^m          — m Grover iterations with adaptive diffusion
            Measure                — collapse to search result

        Parameters
        ----------
        m : int
            Number of Grover iterations.
        alpha : float
            Adaptive diffusion strength.

        Returns
        -------
        QuantumCircuit
            Complete EGAAT search circuit.
        """
        oracle = build_multi_target_oracle(self.n_qubits, self.targets)
        diffusion = adaptive_diffusion(self.n_qubits, alpha)

        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qr, cr, name=f'EGAAT[n={self.n_qubits},m={m},α={alpha:.2f}]')

        # Layer 0: Uniform superposition
        qc.h(qr)

        # Layers 1+3 interleaved: Oracle then adaptive diffusion, m times
        for _ in range(m):
            qc.compose(oracle, qubits=list(range(self.n_qubits)), inplace=True)
            qc.compose(diffusion, qubits=list(range(self.n_qubits)), inplace=True)

        # Measurement
        qc.measure(qr, cr)

        return qc

    def run(self, verbose: bool = False) -> EGAATResult:
        """
        Execute the full EGAAT pipeline and return results.

        Pipeline:
          Step 1 — Oracle construction
          Step 2 — MLAE/QAE to estimate k
          Step 3 — Dynamic iteration count computation
          Step 4 — α scheduling from density estimate
          Step 5 — Execute search circuit
          Step 6 — Decode and return results
        """
        if verbose:
            print(f"[EGAAT] n={self.n_qubits}, N={self.N}, |targets|={len(self.targets)}")

        # Step 1: Build oracle
        oracle = build_multi_target_oracle(self.n_qubits, self.targets)

        # Step 2: Estimate k via QAE/MLAE (without knowing true targets)
        k_est = estimate_k(oracle, self.n_qubits, method=self.estimator_method)
        if verbose:
            print(f"[EGAAT] k_estimated={k_est} (true={len(self.targets)})")

        # Step 3: Dynamic iteration control
        m_opt = optimal_iterations(self.N, k_est)
        if verbose:
            print(f"[EGAAT] Optimal iterations m*={m_opt}")

        # Step 4: Adaptive diffusion coefficient
        alpha = compute_alpha(k_est, self.N)
        if verbose:
            print(f"[EGAAT] Adaptive alpha={alpha:.2f} (density rho={k_est/self.N:.3f})")

        # Step 5: Build and execute search circuit
        qc = self.build_search_circuit(m_opt, alpha)
        transpiled = transpile(qc, self.backend, optimization_level=2)

        result = self.backend.run(transpiled, shots=self.shots).result()
        counts = result.get_counts()

        # Step 6: Decode results (top measurement outcomes)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        found = []
        total_shots = sum(counts.values())

        for bitstring, count in sorted_counts[:10]:
            idx = int(bitstring, 2)
            found.append(idx)

        # Measure success: fraction of shots landing on any true target
        target_shots = sum(counts.get(format(t, f'0{self.n_qubits}b'), 0) for t in self.targets)
        p_measured = target_shots / total_shots

        p_theoretical = success_probability(self.N, len(self.targets), m_opt)

        return EGAATResult(
            found_targets=found,
            true_targets=self.targets,
            n_qubits=self.n_qubits,
            k_estimated=k_est,
            k_true=len(self.targets),
            iterations_used=m_opt,
            alpha_used=alpha,
            success_probability_measured=p_measured,
            success_probability_theoretical=p_theoretical,
            circuit_depth=qc.depth(),
            shot_counts=counts
        )


if __name__ == "__main__":
    print("=== EGAAT Full Pipeline Test ===\n")

    n = 5
    targets = [5, 12, 20, 28]

    egaat = EGAATCircuit(n_qubits=n, targets=targets, shots=4096,
                         estimator_method='qae')

    # Run the full pipeline
    result = egaat.run(verbose=True)

    print(f"\n--- Results ---")
    print(f"True targets:      {sorted(result.true_targets)}")
    print(f"Found (top 5):     {result.found_targets[:5]}")
    print(f"Success:           {result.is_success}")
    print(f"P(success) meas:   {result.success_probability_measured:.4f}")
    print(f"P(success) theory: {result.success_probability_theoretical:.4f}")
    print(f"Circuit depth:     {result.circuit_depth}")
    print(f"Iterations used:   {result.iterations_used}")
    print(f"Alpha used:        {result.alpha_used:.2f}")


def run_with_known_k(n_qubits, targets, shots=4096):
    """Direct run bypassing MLAE — uses true k for clean benchmarking."""
    from src.iteration_control.dynamic_iteration import optimal_iterations, success_probability
    from src.diffusion.adaptive_diffusion import compute_alpha
    from src.oracle.multi_target_oracle import build_multi_target_oracle
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    N = 2 ** n_qubits
    k = len(targets)
    m = optimal_iterations(N, k)
    alpha = compute_alpha(k, N)

    egaat = EGAATCircuit(n_qubits, targets, shots=shots)
    qc = egaat.build_search_circuit(m, alpha)
    backend = AerSimulator(method='statevector')
    transpiled = transpile(qc, backend, optimization_level=2)
    counts = backend.run(transpiled, shots=shots).result().get_counts()

    target_shots = sum(counts.get(format(t, f'0{n_qubits}b'), 0) for t in targets)
    p = target_shots / shots
    p_theory = success_probability(N, k, m)

    print(f"\n=== EGAAT Direct (true k={k}) ===")
    print(f"P(success) measured:   {p:.4f}")
    print(f"P(success) theory:     {p_theory:.4f}")
    print(f"Circuit depth:         {qc.depth()}")
    print(f"Iterations m*:         {m}")
    print(f"Alpha:                 {alpha:.2f}")
    return p


if __name__ == "__main__":
    run_with_known_k(n_qubits=4, targets=[3, 9, 14])
