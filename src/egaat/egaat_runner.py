"""
egaat_runner.py
===============
High-level EGAAT runner for batch experiments and benchmarking.

Wraps EGAATCircuit for running multiple database configurations,
collecting statistics, and comparing against baselines.
"""

import numpy as np
import json
from typing import List, Dict, Optional
from dataclasses import asdict

from egaat_circuit import EGAATCircuit, EGAATResult


def run_single(
    n_qubits: int,
    targets: List[int],
    shots: int = 4096,
    method: str = 'mlae',
    noise_model=None,
    verbose: bool = False
) -> EGAATResult:
    """Run EGAAT on a single configuration."""
    egaat = EGAATCircuit(
        n_qubits=n_qubits,
        targets=targets,
        estimator_method=method,
        shots=shots,
        noise_model=noise_model
    )
    return egaat.run(verbose=verbose)


def run_batch(
    configurations: List[Dict],
    shots: int = 2048,
    method: str = 'mlae',
    noise_model=None
) -> List[EGAATResult]:
    """
    Run EGAAT on multiple (n_qubits, targets) configurations.

    Parameters
    ----------
    configurations : List[Dict]
        Each dict must have keys 'n_qubits' and 'targets'.
    shots : int
    method : str
    noise_model : optional

    Returns
    -------
    List[EGAATResult]
    """
    results = []
    for i, cfg in enumerate(configurations):
        print(f"[{i+1}/{len(configurations)}] n={cfg['n_qubits']}, k={len(cfg['targets'])}")
        r = run_single(
            n_qubits=cfg['n_qubits'],
            targets=cfg['targets'],
            shots=shots,
            method=method,
            noise_model=noise_model
        )
        results.append(r)
        print(f"  P(success)={r.success_probability_measured:.4f}, depth={r.circuit_depth}")
    return results


def summarise(results: List[EGAATResult]) -> Dict:
    """Aggregate statistics across a batch of results."""
    probs = [r.success_probability_measured for r in results]
    depths = [r.circuit_depth for r in results]
    k_errors = [abs(r.k_estimated - r.k_true) for r in results]

    return {
        'n_runs': len(results),
        'mean_success_prob': float(np.mean(probs)),
        'std_success_prob':  float(np.std(probs)),
        'min_success_prob':  float(np.min(probs)),
        'max_success_prob':  float(np.max(probs)),
        'mean_circuit_depth': float(np.mean(depths)),
        'mean_k_error':       float(np.mean(k_errors)),
        'all_above_94_pct':   all(p >= 0.94 for p in probs),
    }


def save_results(results: List[EGAATResult], path: str):
    """Save results to JSON for later analysis."""
    data = []
    for r in results:
        d = {
            'n_qubits': r.n_qubits,
            'k_true': r.k_true,
            'k_estimated': r.k_estimated,
            'iterations': r.iterations_used,
            'alpha': r.alpha_used,
            'p_measured': r.success_probability_measured,
            'p_theoretical': r.success_probability_theoretical,
            'circuit_depth': r.circuit_depth,
            'success': r.is_success
        }
        data.append(d)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} results to {path}")


if __name__ == "__main__":
    import random
    random.seed(42)

    configs = [
        {'n_qubits': 4, 'targets': random.sample(range(16), 2)},
        {'n_qubits': 5, 'targets': random.sample(range(32), 4)},
        {'n_qubits': 6, 'targets': random.sample(range(64), 6)},
    ]

    print("=== EGAAT Batch Runner ===\n")
    results = run_batch(configs, shots=2048)
    summary = summarise(results)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
