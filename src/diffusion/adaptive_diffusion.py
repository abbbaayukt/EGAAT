"""
adaptive_diffusion.py
=====================
EGAAT Core Contribution #3 — Adaptive Diffusion Operator

The standard Grover diffusion D = 2|s><s| - I is applied at full
strength regardless of target density. EGAAT modulates the diffusion
strength based on estimated target density ρ = k/N:

  - Dense search (large ρ): attenuate diffusion — fewer reflections needed
  - Sparse search (small ρ): full diffusion — maximise amplitude transfer

This is implemented as a parameterised diffusion operator D(α) where
α ∈ [0,1] is the amplification coefficient, adjusted per-iteration.

Reference: EGAAT paper §1.2 (third layer), §1.5 (contribution 3)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from typing import List, Optional


def grover_diffusion(n_qubits: int) -> QuantumCircuit:
    """
    Standard Grover diffusion operator: D = 2|s><s| - I
    (inversion about the mean / inversion about uniform superposition).

    Circuit: H^n · (2|0><0| - I) · H^n
    where (2|0><0| - I) is implemented as X^n · MCZ · X^n.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    QuantumCircuit
        Standard diffusion circuit.
    """
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr, name='D_standard')

    # H on all qubits
    qc.h(qr)

    # X on all qubits
    qc.x(qr)

    # Multi-controlled Z (phase -1 on |11...1>)
    qc.h(qr[n_qubits - 1])
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(qr[n_qubits - 1])

    # X on all qubits
    qc.x(qr)

    # H on all qubits
    qc.h(qr)

    return qc


def adaptive_diffusion(n_qubits: int, alpha: float) -> QuantumCircuit:
    """
    Adaptive diffusion operator D(α) parameterised by amplification
    strength α ∈ [0, 1].

    D(α) = I + α · (2|s><s| - 2I) = (1 - 2α)·I + 2α·|s><s|

    At α = 1.0: identical to standard Grover diffusion.
    At α = 0.5: half-strength amplification (for dense targets).
    At α = 0.0: identity (no amplification).

    Implementation: controlled rotation variant of the diffusion.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    alpha : float
        Amplification coefficient in [0, 1].

    Returns
    -------
    QuantumCircuit
        Parameterised diffusion circuit.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha:.4f}")

    if np.isclose(alpha, 1.0, atol=1e-6):
        return grover_diffusion(n_qubits)

    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr, name=f'D_adaptive(α={alpha:.2f})')

    # H on all
    qc.h(qr)
    # X on all
    qc.x(qr)

    # Parameterised phase: instead of full -1 on |11...1>,
    # apply phase e^(iθ) where θ = π·α
    theta = np.pi * alpha

    # Apply multi-controlled phase shift theta on |11...1>
    # This is equivalent to applying phase e^(iθ) to the state |s>
    # since it is surrounded by H and X gates.
    qc.mcp(theta, list(range(n_qubits - 1)), n_qubits - 1)

    # X on all
    qc.x(qr)
    # H on all
    qc.h(qr)

    return qc


def compute_alpha(k_estimate: int, N: int) -> float:
    """
    Compute the adaptive amplification coefficient α based on
    estimated target density ρ = k/N.

    Scheduling strategy (from EGAAT §1.2):
    - ρ < 0.05  (sparse):  α = 1.00  — full amplification
    - ρ < 0.20  (low):     α = 0.85
    - ρ < 0.40  (medium):  α = 0.65
    - ρ < 0.60  (high):    α = 0.45
    - ρ >= 0.60 (dense):   α = 0.25  — attenuated

    This piecewise schedule is the empirically tuned version from
    the benchmark results (Section 4 of EGAAT paper).

    Parameters
    ----------
    k_estimate : int
        Estimated number of target states.
    N : int
        Total database size.

    Returns
    -------
    float
        α value in [0, 1].
    """
    rho = k_estimate / N

    if rho < 0.05:
        return 1.00
    elif rho < 0.20:
        return 0.85
    elif rho < 0.40:
        return 0.65
    elif rho < 0.60:
        return 0.45
    else:
        return 0.25


def continuous_alpha(k_estimate: int, N: int) -> float:
    """
    Alternative: smooth continuous α schedule.
    α(ρ) = 1 - 0.75·ρ

    Simpler formula, slightly lower performance at extremes
    but easier to analyse theoretically.
    """
    rho = min(1.0, k_estimate / N)
    return float(np.clip(1.0 - 0.75 * rho, 0.1, 1.0))


def diffusion_matrix(n_qubits: int, alpha: float = 1.0) -> np.ndarray:
    """
    Classical matrix form of D(α) for analysis and unit testing.

    D(α) = (1 - 2α/N)·I + (2α/N)·J
    where J is the all-ones matrix (outer product of uniform state).
    """
    N = 2 ** n_qubits
    s = np.ones(N) / np.sqrt(N)
    outer = np.outer(s, s)  # |s><s|
    return (1 - 2 * alpha) * np.eye(N) + 2 * alpha * N * outer


def expected_success_after_diffusion(
    n_qubits: int,
    k: int,
    m: int,
    alpha: float
) -> float:
    """
    Theoretical success probability using adaptive diffusion.

    For full α=1 this reduces to the standard Grover formula.
    For α<1 the effective rotation angle per iteration is θ_eff = α·θ.
    """
    N = 2 ** n_qubits
    theta = np.arcsin(np.sqrt(k / N))
    theta_eff = alpha * theta
    return float(np.sin((2 * m + 1) * theta_eff) ** 2)


if __name__ == "__main__":
    print("=== Adaptive Diffusion Operator ===\n")

    N = 256
    test_densities = [1, 5, 30, 80, 150]

    print(f"{'k':>5} {'ρ=k/N':>8} {'α':>6}")
    print("-" * 25)
    for k in test_densities:
        alpha = compute_alpha(k, N)
        rho = k / N
        print(f"{k:>5} {rho:>8.3f} {alpha:>6.2f}")

    print("\n=== Circuit Depth Comparison ===")
    for n in [3, 4, 6, 8]:
        std = grover_diffusion(n)
        ada = adaptive_diffusion(n, alpha=0.65)
        print(f"n={n}: standard depth={std.depth()}, adaptive depth={ada.depth()}")
