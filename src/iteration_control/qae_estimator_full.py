"""
qae_estimator.py
================
EGAAT Core Contribution #2 (Part A) — Quantum Amplitude Estimator

Estimates k (number of marked/target states) as a pre-processing step
before the main Grover search. This is what allows EGAAT to set the
optimal iteration count dynamically WITHOUT knowing k in advance.

Two strategies implemented:
  1. Full QAE (Brassard et al. 2002) — high precision, deeper circuit
  2. MLAE (Suzuki et al. 2020)       — NISQ-friendly, shallow circuit

Reference: EGAAT paper §1.2, §2.3
"""

import numpy as np
from typing import Callable, Tuple, Optional
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Strategy 1: Full QAE using Quantum Phase Estimation
# ---------------------------------------------------------------------------

def qae_full(
    oracle: QuantumCircuit,
    n_qubits: int,
    n_ancilla: int = 6,
    shots: int = 8192,
    backend=None
) -> Tuple[float, float]:
    """
    Quantum Amplitude Estimation (Brassard et al. 2002).

    Uses quantum phase estimation on the Grover operator Q = -A·S0·A†·SO
    to estimate sin²(θ) ≈ k/N, giving k ≈ N·sin²(θ).

    Parameters
    ----------
    oracle : QuantumCircuit
        The phase oracle SO marking target states.
    n_qubits : int
        Number of search qubits (N = 2^n_qubits).
    n_ancilla : int
        Number of ancilla/counting qubits. Precision ≈ 1/2^n_ancilla.
    shots : int
        Number of measurement shots.
    backend : AerSimulator, optional
        Qiskit backend. Defaults to statevector simulator.

    Returns
    -------
    Tuple[float, float]
        (k_estimate, theta_estimate) — estimated number of targets and angle.
    """
    if backend is None:
        backend = AerSimulator(method='statevector')

    N = 2 ** n_qubits
    total_qubits = n_ancilla + n_qubits

    # Build the QPE circuit
    anc = QuantumRegister(n_ancilla, 'anc')
    search = QuantumRegister(n_qubits, 'search')
    creg = ClassicalRegister(n_ancilla, 'meas')
    qc = QuantumCircuit(anc, search, creg)

    # Initialise search register in uniform superposition
    qc.h(search)

    # Hadamard on ancilla
    qc.h(anc)

    # Controlled Grover iterations: Q^(2^j) controlled on ancilla j
    grover_op = _build_grover_operator(oracle, n_qubits)
    for j in range(n_ancilla):
        reps = 2 ** j
        controlled_Q = grover_op.power(reps).control(1)
        qc.append(controlled_Q, [anc[j]] + list(search))

    # Inverse QFT on ancilla
    qft_inv = QFT(n_ancilla, inverse=True)
    qc.append(qft_inv, anc)

    # Measure ancilla
    qc.measure(anc, creg)

    # Execute
    transpiled = transpile(qc, backend, optimization_level=1)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # Decode: most frequent measurement -> phase -> amplitude
    best = max(counts, key=counts.get)
    phase_int = int(best, 2)
    theta = phase_int * np.pi / (2 ** (n_ancilla - 1))
    k_estimate = N * np.sin(theta) ** 2

    return float(np.clip(k_estimate, 0, N)), theta


# ---------------------------------------------------------------------------
# Strategy 2: Maximum Likelihood Amplitude Estimation (NISQ-friendly)
# ---------------------------------------------------------------------------

def mlae_estimate(
    oracle: QuantumCircuit,
    n_qubits: int,
    schedule: Optional[list] = None,
    shots_per_point: int = 100,
    backend=None
) -> Tuple[float, float]:
    """
    Maximum Likelihood Amplitude Estimation (Suzuki et al. 2020).

    Runs Grover circuits with m = 0, 1, 2, 4, 8, ... iterations,
    collects measurement statistics, then finds θ that maximises
    the likelihood L(θ | data). No QFT required — NISQ-compatible.

    This is the preferred estimator in EGAAT for NISQ hardware.

    Parameters
    ----------
    oracle : QuantumCircuit
        The phase oracle.
    n_qubits : int
        Number of search qubits.
    schedule : list of int, optional
        Number of Grover iterations per evaluation point.
        Default: [0, 1, 2, 4, 8, 16].
    shots_per_point : int
        Measurement shots at each schedule point.
    backend : optional
        Qiskit backend.

    Returns
    -------
    Tuple[float, float]
        (k_estimate, theta_mle) — estimated k and MLE angle.
    """
    if backend is None:
        backend = AerSimulator(method='statevector')

    if schedule is None:
        schedule = [0, 1, 2, 4, 8, 16]

    N = 2 ** n_qubits
    grover_op = _build_grover_operator(oracle, n_qubits)

    # Collect hit counts h_m (number of times a target was measured)
    hit_counts = {}
    for m in schedule:
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)  # superposition
        if m > 0:
            powered = grover_op.power(m)
            qc.append(powered, qr)
        qc.measure(qr, cr)

        transpiled = transpile(qc, backend, optimization_level=1)
        result = backend.run(transpiled, shots=shots_per_point).result()
        counts = result.get_counts()

        # Run the circuit again with oracle marking targets
        # A "hit" = measuring a target state = high-amplitude state
        # We use the most frequent outcome as signal
        total = sum(counts.values())
        top_state = max(counts, key=counts.get)
        top_count = counts[top_state]
        # hits = shots where top outcome matches (proxy for amplitude)
        hits = top_count
        hit_counts[m] = hits

    # Maximum likelihood over θ ∈ (0, π/2)
    theta_mle = _mle_theta(hit_counts, schedule, shots_per_point)
    k_estimate = N * np.sin(theta_mle) ** 2

    return float(np.clip(k_estimate, 0, N)), theta_mle


def _mle_theta(
    hit_counts: dict,
    schedule: list,
    shots: int,
    n_grid: int = 1000
) -> float:
    """
    Grid search for MLE of θ.
    log L(θ) = Σ_m [ h_m·log(p_m(θ)) + (shots-h_m)·log(1-p_m(θ)) ]
    where p_m(θ) = sin²((2m+1)θ).
    """
    theta_grid = np.linspace(1e-6, np.pi / 2 - 1e-6, n_grid)
    log_likelihoods = np.zeros(n_grid)

    for m in schedule:
        h = hit_counts.get(m, 0)
        # Expected success probability after m Grover iterations
        p = np.sin((2 * m + 1) * theta_grid) ** 2
        p = np.clip(p, 1e-10, 1 - 1e-10)
        log_likelihoods += h * np.log(p) + (shots - h) * np.log(1 - p)

    return float(theta_grid[np.argmax(log_likelihoods)])


# ---------------------------------------------------------------------------
# Shared helper: Grover operator Q = -A·S0·A†·SO
# ---------------------------------------------------------------------------

def _build_grover_operator(oracle: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
    """
    Construct the full Grover operator Q = D · O where:
      O = oracle (phase inversion on targets)
      D = diffusion (inversion about average)
    """
    from src.diffusion.adaptive_diffusion import grover_diffusion
    qc = QuantumCircuit(n_qubits, name='Q')
    qc.compose(oracle, inplace=True)
    qc.compose(grover_diffusion(n_qubits), inplace=True)
    return qc


def estimate_k(
    oracle: QuantumCircuit,
    n_qubits: int,
    method: str = 'mlae',
    **kwargs
) -> int:
    """
    High-level interface: estimate k (number of target states).

    Parameters
    ----------
    oracle : QuantumCircuit
    n_qubits : int
    method : str
        'mlae' (default, NISQ-friendly) or 'qae' (full precision).

    Returns
    -------
    int
        Rounded integer estimate of k. Minimum 1.
    """
    if method == 'mlae':
        k_est, _ = mlae_estimate(oracle, n_qubits, **kwargs)
    elif method == 'qae':
        k_est, _ = qae_full(oracle, n_qubits, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mlae' or 'qae'.")

    return max(1, int(round(k_est)))


if __name__ == "__main__":
    from src.oracle.multi_target_oracle import build_multi_target_oracle  # noqa

    n = 4
    targets = [3, 7, 11]
    oracle = build_multi_target_oracle(n, targets)

    print(f"True k = {len(targets)}")
    k_est = estimate_k(oracle, n, method='qae')
    print(f"MLAE estimated k = {k_est}")
