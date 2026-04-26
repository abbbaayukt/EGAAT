"""
constraint_satisfaction.py
===========================
Application: EGAAT for Constraint Satisfaction Problems (CSP)

Models a simple binary CSP where solutions must satisfy a set of
clauses. EGAAT searches for all satisfying assignments simultaneously.

Reference: EGAAT §1.4 (optimisation application)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from typing import List, Callable


def build_csp_oracle_targets(
    n_vars: int,
    clauses: List[Callable[[int], bool]]
) -> List[int]:
    """
    Find all variable assignments satisfying all clauses.

    Parameters
    ----------
    n_vars : int
        Number of binary variables. Search space = 2^n_vars.
    clauses : List[Callable[[int], bool]]
        Each clause takes an integer assignment and returns True if satisfied.

    Returns
    -------
    List[int]
        All satisfying assignments (indices in the search space).
    """
    N = 2 ** n_vars
    return [x for x in range(N) if all(c(x) for c in clauses)]


def run_csp_search(n_vars: int = 5):
    """
    Demonstrate EGAAT solving a small 3-SAT-like CSP.

    Example: Find all 5-bit assignments where:
      - Bit 0 AND Bit 1 = 1 (both must be 1)
      - Bit 2 OR Bit 3 = 1 (at least one must be 1)
      - NOT Bit 4 = 1 (bit 4 must be 0)
    """
    from src.egaat.egaat_circuit import EGAATCircuit

    clauses = [
        lambda x: bool(x & 0b00001) and bool(x & 0b00010),  # bits 0 AND 1
        lambda x: bool(x & 0b00100) or bool(x & 0b01000),   # bits 2 OR 3
        lambda x: not bool(x & 0b10000)                       # NOT bit 4
    ]

    targets = build_csp_oracle_targets(n_vars, clauses)
    N = 2 ** n_vars

    print(f"CSP Search: {n_vars} variables, 3 clauses")
    print(f"  Satisfying assignments: {len(targets)}/{N} ({len(targets)/N*100:.1f}%)")
    print(f"  Assignments: {[bin(t) for t in targets]}")

    if not targets:
        print("  No solutions exist.")
        return

    egaat = EGAATCircuit(n_vars, targets, shots=2048)
    result = egaat.run()

    print(f"  EGAAT P(success): {result.success_probability_measured:.4f}")
    print(f"  Found: {[bin(t) for t in result.found_targets[:5]]}")
    return result


if __name__ == "__main__":
    print("=== EGAAT: Constraint Satisfaction ===\n")
    run_csp_search(n_vars=5)
