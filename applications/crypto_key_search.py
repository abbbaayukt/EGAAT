"""
crypto_key_search.py
====================
Application: EGAAT for Cryptographic Key Search

Demonstrates EGAAT's application to searching for keys satisfying
a partial constraint (e.g. finding all keys whose first n bits match
a known pattern — relevant in differential cryptanalysis).

This is a simulation of the quantum advantage scenario, not a real
attack on any cipher. Database = space of possible key fragments.

Reference: EGAAT §1.4 (motivation — cryptography application)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from typing import List, Tuple


def simulate_partial_key_search(
    key_bits: int = 6,
    pattern_bits: int = 2,
    pattern: int = 0b11,
) -> dict:
    """
    Search for all keys whose top `pattern_bits` match `pattern`.

    In a real quantum setting, the oracle would be a reversible
    implementation of the partial-match predicate.
    Here we simulate it classically to demonstrate the setup.

    Parameters
    ----------
    key_bits : int
        Total key fragment size in bits (database size N = 2^key_bits).
    pattern_bits : int
        Number of constrained bits.
    pattern : int
        The target bit pattern to match.

    Returns
    -------
    dict
        Analysis including EGAAT parameters, theoretical speedup.
    """
    from src.egaat.egaat_circuit import EGAATCircuit
    from src.iteration_control.dynamic_iteration import optimal_iterations, query_complexity

    N = 2 ** key_bits
    mask = (2 ** pattern_bits - 1) << (key_bits - pattern_bits)
    shifted_pattern = pattern << (key_bits - pattern_bits)

    # Find all keys matching the pattern (these are the "targets")
    targets = [k for k in range(N) if (k & mask) == shifted_pattern]
    k = len(targets)  # Theoretically N / 2^pattern_bits

    m_star = optimal_iterations(N, k)
    classical_avg = N // 2
    quantum_queries = query_complexity(N, k)
    speedup = classical_avg / quantum_queries

    print(f"Key Search Setup:")
    print(f"  Key fragment bits: {key_bits} → N={N} candidates")
    print(f"  Pattern constraint: top {pattern_bits} bits = {pattern:0{pattern_bits}b}")
    print(f"  Matching keys: {k} ({k/N*100:.1f}% of space)")
    print(f"  Classical queries: ~{classical_avg}")
    print(f"  EGAAT queries:     ~{quantum_queries:.1f}  (m*={m_star})")
    print(f"  Quantum speedup:   {speedup:.2f}x")

    # Run EGAAT
    egaat = EGAATCircuit(key_bits, targets, shots=2048)
    result = egaat.run()

    return {
        'N': N, 'k': k,
        'classical_queries': classical_avg,
        'quantum_queries': quantum_queries,
        'speedup': speedup,
        'egaat_p_success': result.success_probability_measured,
        'found': result.found_targets[:5]
    }


if __name__ == "__main__":
    print("=== EGAAT: Cryptographic Key Search Application ===\n")
    r = simulate_partial_key_search(key_bits=6, pattern_bits=2, pattern=0b11)
    print(f"\nEGAAT P(success) = {r['egaat_p_success']:.4f}")
    print(f"Sample found keys: {r['found']}")
    print(f"True targets (first 5): {[k for k in range(r['N']) if k >> 4 == 3][:5]}")
