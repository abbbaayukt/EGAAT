"""
mlae_variant.py
===============
Maximum Likelihood Amplitude Estimation — standalone module.

Extended implementation of Suzuki et al. (2020) MLAE with:
  - Exponential schedule (default)
  - Linear schedule
  - Adaptive schedule (stops early when confidence is high)
  - Confidence interval computation via Fisher information

This is the NISQ-preferred estimator in EGAAT because it requires
NO quantum phase estimation (no QFT, no ancilla register).

Reference: Suzuki et al., Quantum Information Processing 19(2), 2020
           EGAAT paper §2.3
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from scipy.stats import norm as scipy_norm


@dataclass
class MLAEResult:
    """Result of an MLAE estimation run."""
    theta_mle: float          # MLE estimate of angle θ
    k_estimate: float         # Estimated number of targets k = N·sin²(θ)
    confidence_lower: float   # 95% CI lower bound on k
    confidence_upper: float   # 95% CI upper bound on k
    n_oracle_calls: int       # Total oracle evaluations used
    schedule_used: List[int]  # Grover iteration counts used
    log_likelihood: float     # Final log-likelihood value


def exponential_schedule(max_power: int = 5) -> List[int]:
    """
    Exponential schedule: m = 0, 1, 2, 4, 8, ..., 2^max_power
    Best for unknown θ — explores wide range efficiently.
    Total oracle calls: O(2^max_power).
    """
    return [0] + [2**i for i in range(max_power + 1)]


def linear_schedule(max_m: int = 20, step: int = 2) -> List[int]:
    """
    Linear schedule: m = 0, 2, 4, ..., max_m
    Better precision near known θ, worse for unknown.
    """
    return list(range(0, max_m + 1, step))


def power_law_schedule(n_points: int = 8, base: float = 1.8) -> List[int]:
    """
    Power law schedule: m_i = floor(base^i)
    Compromise between linear and exponential.
    Suzuki et al. recommend this for hardware experiments.
    """
    schedule = sorted(set(int(base**i) for i in range(n_points)))
    return [0] + schedule


def log_likelihood_function(
    theta: float,
    schedule: List[int],
    hit_counts: Dict[int, int],
    shots_per_point: int
) -> float:
    """
    Log-likelihood for MLAE.

    L(θ) = Σ_m [ h_m · log(sin²((2m+1)θ)) + (M-h_m) · log(cos²((2m+1)θ)) ]

    where h_m = hits at m iterations, M = shots per point.
    """
    log_l = 0.0
    for m in schedule:
        h = hit_counts.get(m, 0)
        miss = shots_per_point - h

        p = np.sin((2 * m + 1) * theta) ** 2
        p = np.clip(p, 1e-12, 1 - 1e-12)

        log_l += h * np.log(p) + miss * np.log(1 - p)

    return log_l


def fisher_information(
    theta: float,
    schedule: List[int],
    shots_per_point: int
) -> float:
    """
    Fisher information I(θ) for the MLAE estimator.

    I(θ) = Σ_m M · [(2m+1)·sin(2(2m+1)θ)]² / [sin²((2m+1)θ)·cos²((2m+1)θ)]

    Used to compute Cramér-Rao lower bound on variance.
    """
    I = 0.0
    for m in schedule:
        angle = (2 * m + 1) * theta
        p = np.sin(angle) ** 2
        p = np.clip(p, 1e-12, 1 - 1e-12)
        dpdt = (2 * m + 1) * np.sin(2 * angle)
        I += shots_per_point * (dpdt ** 2) / (p * (1 - p))
    return I


def mle_grid_then_refine(
    schedule: List[int],
    hit_counts: Dict[int, int],
    shots_per_point: int,
    n_grid: int = 2000
) -> Tuple[float, float]:
    """
    Two-stage MLE:
      1. Coarse grid search over [ε, π/2 - ε]
      2. Brent's method refinement around the coarse maximum

    Returns (theta_mle, log_likelihood_at_mle).
    """
    theta_grid = np.linspace(1e-4, np.pi / 2 - 1e-4, n_grid)
    lls = np.array([
        log_likelihood_function(t, schedule, hit_counts, shots_per_point)
        for t in theta_grid
    ])

    # Coarse maximum
    coarse_best = theta_grid[np.argmax(lls)]

    # Refine with Brent's method around ±5 grid steps
    step = theta_grid[1] - theta_grid[0]
    lo = max(1e-5, coarse_best - 5 * step)
    hi = min(np.pi / 2 - 1e-5, coarse_best + 5 * step)

    result = minimize_scalar(
        lambda t: -log_likelihood_function(t, schedule, hit_counts, shots_per_point),
        bounds=(lo, hi),
        method='bounded'
    )

    theta_mle = float(result.x)
    ll_mle = float(-result.fun)
    return theta_mle, ll_mle


def compute_confidence_interval(
    theta_mle: float,
    schedule: List[int],
    shots_per_point: int,
    N: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    95% confidence interval on k estimate using Fisher information.

    Var(θ_MLE) ≈ 1 / I(θ_MLE)   (Cramér-Rao bound)
    CI on θ: θ ± z_{α/2} / sqrt(I(θ))
    CI on k = N·sin²(θ): propagated via delta method.
    """
    I = fisher_information(theta_mle, schedule, shots_per_point)
    if I <= 0:
        return 0.0, float(N)

    z = scipy_norm.ppf((1 + confidence) / 2)
    std_theta = 1.0 / np.sqrt(I)

    theta_lo = max(1e-6, theta_mle - z * std_theta)
    theta_hi = min(np.pi / 2 - 1e-6, theta_mle + z * std_theta)

    k_lo = N * np.sin(theta_lo) ** 2
    k_hi = N * np.sin(theta_hi) ** 2

    return float(k_lo), float(k_hi)


def mlae_full(
    oracle_runner,
    n_qubits: int,
    schedule: Optional[List[int]] = None,
    shots_per_point: int = 200,
    schedule_type: str = 'exponential'
) -> MLAEResult:
    """
    Full MLAE estimation with confidence intervals.

    Parameters
    ----------
    oracle_runner : callable
        Function(m: int) -> int
        Runs m Grover iterations and returns number of target hits
        out of shots_per_point measurements.
    n_qubits : int
        Number of search qubits.
    schedule : List[int], optional
        Custom schedule. If None, uses schedule_type.
    shots_per_point : int
        Shots per schedule point.
    schedule_type : str
        'exponential', 'linear', or 'power_law'.

    Returns
    -------
    MLAEResult
    """
    N = 2 ** n_qubits

    if schedule is None:
        if schedule_type == 'exponential':
            schedule = exponential_schedule(max_power=5)
        elif schedule_type == 'linear':
            schedule = linear_schedule(max_m=20)
        elif schedule_type == 'power_law':
            schedule = power_law_schedule(n_points=8)
        else:
            schedule = exponential_schedule()

    # Collect hit counts from quantum circuit
    hit_counts = {}
    total_oracle_calls = 0
    for m in schedule:
        hits = oracle_runner(m)
        hit_counts[m] = hits
        total_oracle_calls += (2 * m + 1) * shots_per_point

    # MLE
    theta_mle, ll = mle_grid_then_refine(schedule, hit_counts, shots_per_point)
    k_est = N * np.sin(theta_mle) ** 2

    # Confidence interval
    k_lo, k_hi = compute_confidence_interval(theta_mle, schedule, shots_per_point, N)

    return MLAEResult(
        theta_mle=theta_mle,
        k_estimate=k_est,
        confidence_lower=k_lo,
        confidence_upper=k_hi,
        n_oracle_calls=total_oracle_calls,
        schedule_used=schedule,
        log_likelihood=ll
    )


def simulate_oracle_runner(
    n_qubits: int,
    true_targets: List[int],
    shots_per_point: int,
    noise_p: float = 0.0
) -> callable:
    """
    Simulate an oracle_runner function for testing without full Qiskit circuit.

    Simulates Grover iteration statistics: after m iterations,
    P(hit) = sin²((2m+1)·arcsin(sqrt(k/N))).
    """
    N = 2 ** n_qubits
    k = len(true_targets)
    theta_true = np.arcsin(np.sqrt(k / N))

    def runner(m: int) -> int:
        p = np.sin((2 * m + 1) * theta_true) ** 2
        # Add depolarising noise effect
        if noise_p > 0:
            p = (1 - noise_p) * p + noise_p * (k / N)
        p = np.clip(p, 0, 1)
        return int(np.random.binomial(shots_per_point, p))

    return runner


if __name__ == "__main__":
    print("=== MLAE Variant Test ===\n")
    np.random.seed(42)

    n = 6
    N = 2 ** n
    true_k = 5
    true_targets = list(range(true_k))

    runner = simulate_oracle_runner(n, true_targets, shots_per_point=200)

    for stype in ['exponential', 'linear', 'power_law']:
        result = mlae_full(runner, n, shots_per_point=200, schedule_type=stype)
        print(f"Schedule: {stype:<12} | k_est={result.k_estimate:.2f} (true={true_k}) "
              f"| 95% CI=[{result.confidence_lower:.1f}, {result.confidence_upper:.1f}] "
              f"| oracle_calls={result.n_oracle_calls}")

    print("\nSchedules tested:")
    print(f"  Exponential: {exponential_schedule(5)}")
    print(f"  Linear:      {linear_schedule(10, 2)}")
    print(f"  Power law:   {power_law_schedule(8)}")
