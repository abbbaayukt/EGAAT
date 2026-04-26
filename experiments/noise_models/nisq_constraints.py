"""
nisq_constraints.py
===================
NISQ hardware constraint modelling for EGAAT validation.

Models realistic NISQ device limitations:
  - Limited qubit connectivity (linear, T-shaped, grid)
  - Coherence time constraints (T1, T2)
  - Gate error rate bounds
  - Maximum viable circuit depth given noise

Used to argue EGAAT is deployable on near-term hardware (§1.4, §2.5).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class NISQProfile:
    """Hardware profile for a NISQ device."""
    name: str
    n_qubits: int
    t1_us: float           # T1 relaxation time (microseconds)
    t2_us: float           # T2 dephasing time (microseconds)
    gate_1q_ns: float      # Single-qubit gate time (nanoseconds)
    gate_2q_ns: float      # Two-qubit gate time (nanoseconds)
    gate_1q_error: float   # Single-qubit error rate
    gate_2q_error: float   # Two-qubit error rate
    connectivity: str      # 'linear', 'grid', 'heavy_hex', 'all'


# Representative NISQ profiles (based on published IBM/Google specs ca. 2024)
IBM_FALCON = NISQProfile(
    name='IBM Falcon (7-qubit)',
    n_qubits=7,
    t1_us=100.0, t2_us=120.0,
    gate_1q_ns=35.0, gate_2q_ns=300.0,
    gate_1q_error=3e-4, gate_2q_error=8e-3,
    connectivity='heavy_hex'
)

IBM_EAGLE = NISQProfile(
    name='IBM Eagle (127-qubit)',
    n_qubits=127,
    t1_us=150.0, t2_us=130.0,
    gate_1q_ns=35.0, gate_2q_ns=320.0,
    gate_1q_error=2e-4, gate_2q_error=6e-3,
    connectivity='heavy_hex'
)

GENERIC_NISQ = NISQProfile(
    name='Generic NISQ (simulated)',
    n_qubits=20,
    t1_us=80.0, t2_us=100.0,
    gate_1q_ns=50.0, gate_2q_ns=300.0,
    gate_1q_error=1e-3, gate_2q_error=1e-2,
    connectivity='linear'
)


def max_viable_depth(profile: NISQProfile, target_fidelity: float = 0.5) -> int:
    """
    Estimate maximum circuit depth before fidelity drops below threshold.

    Uses: F ≈ (1 - p_2q)^(n_2q_gates) where p_2q is two-qubit error rate.
    Solves for n_2q_gates such that F ≥ target_fidelity.

    Returns
    -------
    int
        Maximum number of two-qubit gate layers.
    """
    if profile.gate_2q_error <= 0:
        return 10000
    max_gates = int(np.log(target_fidelity) / np.log(1 - profile.gate_2q_error))
    return max(1, max_gates)


def egaat_circuit_depth(n_qubits: int, k: int) -> int:
    """
    Estimate EGAAT circuit depth for given n and k.

    Depth ≈ m* · (oracle_depth + diffusion_depth)
    oracle_depth ≈ n_qubits * 4   (log-depth oracle)
    diffusion_depth ≈ n_qubits * 4
    """
    from src.iteration_control.dynamic_iteration import optimal_iterations
    N = 2 ** n_qubits
    m = optimal_iterations(N, k)
    per_iteration = n_qubits * 4 * 2  # oracle + diffusion
    return m * per_iteration


def is_nisq_feasible(
    n_qubits: int,
    k: int,
    profile: NISQProfile,
    target_fidelity: float = 0.5
) -> dict:
    """
    Check whether EGAAT is feasible on a given NISQ device.

    Returns analysis dict with feasibility flag and metrics.
    """
    depth = egaat_circuit_depth(n_qubits, k)
    max_depth = max_viable_depth(profile, target_fidelity)

    # Coherence check: total circuit time must be < T2
    gate_time_ns = (depth // 2) * profile.gate_1q_ns + (depth // 2) * profile.gate_2q_ns
    gate_time_us = gate_time_ns / 1000

    feasible = (depth <= max_depth) and (gate_time_us < profile.t2_us)

    return {
        'device': profile.name,
        'n_qubits': n_qubits,
        'k': k,
        'egaat_depth': depth,
        'max_viable_depth': max_depth,
        'circuit_time_us': round(gate_time_us, 2),
        'coherence_time_us': profile.t2_us,
        'feasible': feasible,
        'depth_utilisation': round(depth / max_depth, 3)
    }


if __name__ == "__main__":
    print("=== NISQ Feasibility Analysis for EGAAT ===\n")

    configs = [(4, 2), (6, 4), (8, 8), (10, 16)]
    profiles = [GENERIC_NISQ, IBM_FALCON, IBM_EAGLE]

    for n, k in configs:
        print(f"\nn={n} qubits, k={k} targets:")
        for profile in profiles:
            result = is_nisq_feasible(n, k, profile)
            status = "✓ FEASIBLE" if result['feasible'] else "✗ TOO DEEP"
            print(f"  [{status}] {profile.name}: "
                  f"depth={result['egaat_depth']}/{result['max_viable_depth']}, "
                  f"time={result['circuit_time_us']}µs/{profile.t2_us}µs")
