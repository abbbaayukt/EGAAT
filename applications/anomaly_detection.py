"""
anomaly_detection.py
====================
Application: EGAAT for Anomaly Detection in Large-Scale Data

Models searching a database of encoded data records for anomalies
(records satisfying some anomaly predicate). Anomalies are rare
(sparse targets) — exactly the regime where EGAAT shines.

Reference: EGAAT §1.4 (anomaly detection application)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import random
from typing import List


def simulate_anomaly_database(
    n_qubits: int,
    anomaly_rate: float = 0.03,
    seed: int = 42
) -> List[int]:
    """
    Simulate a database where ~anomaly_rate fraction of records are anomalous.
    Returns list of anomalous record indices (the search targets).
    """
    random.seed(seed)
    N = 2 ** n_qubits
    k = max(1, int(N * anomaly_rate))
    return random.sample(range(N), k)


def run_anomaly_search(n_qubits: int = 7, anomaly_rate: float = 0.03):
    """
    Run EGAAT to detect anomalies in a simulated dataset.

    In the sparse regime (small k/N), EGAAT provides maximum quantum
    advantage because m* = O(sqrt(N/k)) >> O(1) for small k.
    """
    from src.egaat.egaat_circuit import EGAATCircuit
    from src.iteration_control.dynamic_iteration import query_complexity

    targets = simulate_anomaly_database(n_qubits, anomaly_rate)
    N = 2 ** n_qubits
    k = len(targets)

    classical_queries = N / 2
    quantum_queries = query_complexity(N, k)
    speedup = classical_queries / quantum_queries

    print(f"Anomaly Detection: N={N} records, k={k} anomalies ({anomaly_rate*100:.1f}%)")
    print(f"  Classical avg queries:  {classical_queries:.0f}")
    print(f"  EGAAT quantum queries:  {quantum_queries:.1f}")
    print(f"  Quantum speedup:        {speedup:.1f}x")

    egaat = EGAATCircuit(n_qubits, targets, shots=2048)
    result = egaat.run(verbose=False)

    # Check how many anomalies were found
    detected = [t for t in result.found_targets if t in targets]
    detection_rate = len(detected) / max(len(targets), 1)

    print(f"\n  P(success):       {result.success_probability_measured:.4f}")
    print(f"  Anomalies found:  {len(detected)}/{k} (detection rate {detection_rate:.2f})")
    return result


if __name__ == "__main__":
    print("=== EGAAT: Anomaly Detection Application ===\n")

    for rate in [0.01, 0.03, 0.10]:
        print(f"\n--- Anomaly rate: {rate*100:.0f}% ---")
        run_anomaly_search(n_qubits=6, anomaly_rate=rate)
