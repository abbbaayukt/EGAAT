"""
dynamic_iteration.py
====================
EGAAT Core Contribution #2 (Part B) — Dynamic Iteration Control

Given an estimate of k (from QAE/MLAE), compute the optimal number
of Grover iterations to maximise success probability WITHOUT over-rotation.

This is the key fix for the standard Grover limitation: without knowing k,
fixed iteration counts cause over/under-rotation, degrading P(success).

Reference: EGAAT paper §1.1 (problem), §1.2 (solution), §2.2 (Yoder comparison)
"""

import numpy as np
from typing import Tuple, Optional


def optimal_iterations(N: int, k: int) -> int:
    """
    Compute the optimal number of Grover iterations given N and k.

    Standard formula: m* = floor(π/4 · sqrt(N/k))

    This maximises P(success) = sin²((2m+1)·θ) where θ = arcsin(√(k/N)).
    At m = m*, this probability is close to 1 for any k.

    Parameters
    ----------
    N : int
        Database size (must be power of 2).
    k : int
        Number of target states (estimated by QAE/MLAE).

    Returns
    -------
    int
        Optimal number of Grover iterations m*.

    Raises
    ------
    ValueError
        If k <= 0 or k > N.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > N:
        raise ValueError(f"k ({k}) cannot exceed N ({N})")

    if k == N:
        return 0  # All states are targets, no search needed

    theta = np.arcsin(np.sqrt(k / N))
    m_star = int(np.floor(np.pi / (4 * theta)))
    return max(1, m_star)


def success_probability(N: int, k: int, m: int) -> float:
    """
    Theoretical success probability after exactly m Grover iterations.

    P(success | m, k, N) = sin²((2m + 1) · arcsin(√(k/N)))

    Parameters
    ----------
    N : int
        Database size.
    k : int
        True number of targets.
    m : int
        Number of iterations applied.

    Returns
    -------
    float
        Success probability in [0, 1].
    """
    theta = np.arcsin(np.sqrt(k / N))
    return float(np.sin((2 * m + 1) * theta) ** 2)


def adaptive_iteration_schedule(
    N: int,
    k_estimate: int,
    safety_margin: float = 0.05
) -> Tuple[int, float]:
    """
    Compute iteration count with a safety margin to guard against
    estimation error in k. Slightly conservative to avoid over-rotation.

    If our estimate of k is off by ±safety_margin fraction, this
    finds the iteration count that maximises the worst-case probability.

    Parameters
    ----------
    N : int
        Database size.
    k_estimate : int
        Estimated number of targets from QAE/MLAE.
    safety_margin : float
        Fractional uncertainty in k estimate (default 5%).

    Returns
    -------
    Tuple[int, float]
        (m_safe, expected_prob) — safe iteration count and expected P(success).
    """
    # Consider k range within safety margin
    k_low  = max(1, int(k_estimate * (1 - safety_margin)))
    k_high = min(N - 1, int(k_estimate * (1 + safety_margin)))

    best_m = 1
    best_worst_case = 0.0

    # Search over candidate iteration counts
    m_max = optimal_iterations(N, k_low) + 5  # upper bound

    for m in range(1, m_max + 1):
        # Worst-case probability across k range
        probs = [
            success_probability(N, k, m)
            for k in range(k_low, k_high + 1)
        ]
        worst = min(probs)
        if worst > best_worst_case:
            best_worst_case = worst
            best_m = m

    return best_m, best_worst_case


def iteration_sweep(N: int, k: int, m_range: Optional[range] = None) -> dict:
    """
    Compute success probability for a range of iteration counts.
    Used to plot the rotation curve and visualise over/under-rotation.

    Parameters
    ----------
    N : int
        Database size.
    k : int
        True number of targets.
    m_range : range, optional
        Iteration counts to sweep. Default: 0 to 3*m*.

    Returns
    -------
    dict
        {'iterations': [...], 'probabilities': [...], 'optimal_m': m*}
    """
    m_star = optimal_iterations(N, k)
    if m_range is None:
        m_range = range(0, 3 * m_star + 1)

    iters = list(m_range)
    probs = [success_probability(N, k, m) for m in iters]

    return {
        'iterations': iters,
        'probabilities': probs,
        'optimal_m': m_star,
        'peak_probability': success_probability(N, k, m_star)
    }


def query_complexity(N: int, k: int) -> float:
    """
    Theoretical query complexity: O(sqrt(N/k)).
    Returns the exact constant-free expression sqrt(N/k).

    This matches the EGAAT abstract claim: O(√(N/k)).
    """
    return np.sqrt(N / k)


if __name__ == "__main__":
    import json

    print("=== EGAAT Dynamic Iteration Control ===\n")

    test_cases = [
        (64,   1),
        (64,   4),
        (256,  8),
        (1024, 16),
        (4096, 64),
    ]

    print(f"{'N':>6} {'k':>5} {'m*':>5} {'P(success)':>12} {'Query Complexity':>18}")
    print("-" * 55)
    for N, k in test_cases:
        m = optimal_iterations(N, k)
        p = success_probability(N, k, m)
        qc = query_complexity(N, k)
        print(f"{N:>6} {k:>5} {m:>5} {p:>12.4f} {qc:>18.2f}")

    print("\n=== Safety Margin Demo (k estimated with 10% error) ===")
    N, k_true = 256, 10
    k_est = 11  # slightly off
    m_safe, worst_prob = adaptive_iteration_schedule(N, k_est, safety_margin=0.10)
    p_naive = success_probability(N, k_true, optimal_iterations(N, k_est))
    print(f"True k={k_true}, Estimated k={k_est}")
    print(f"Naive m* = {optimal_iterations(N, k_est)}, P = {p_naive:.4f}")
    print(f"Safe  m  = {m_safe}, worst-case P = {worst_prob:.4f}")
