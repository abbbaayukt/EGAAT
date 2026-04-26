"""
run_all_experiments.py
======================
Master runner: execute all EGAAT experiments end to end.

Run this single file to reproduce all results from the paper.
Estimated wall time: 15–45 min depending on hardware.

Usage:
    python experiments/run_all_experiments.py
    python experiments/run_all_experiments.py --quick   # fast subset
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmarks.db_64   import run as run_db64
from experiments.benchmarks.db_256  import run as run_db256
from experiments.comparisons.success_probability import run_comparison
from experiments.comparisons.query_complexity    import measure_query_complexity
from experiments.comparisons.circuit_depth       import measure_depths
from experiments.noise_models.nisq_constraints   import is_nisq_feasible, GENERIC_NISQ


def run_all(quick: bool = False):
    results = {}
    t0 = time.time()

    print("=" * 60)
    print("EGAAT — Full Experimental Suite")
    print("=" * 60)

    # 1. Small database benchmarks (always run)
    print("\n[1/5] N=64 Benchmark...")
    results['db64'] = run_db64()

    if not quick:
        print("\n[2/5] N=256 Benchmark...")
        results['db256'] = run_db256()
    else:
        print("\n[2/5] Skipped (--quick)")

    # 3. Success probability comparison
    print("\n[3/5] Success probability comparison (N=16, quick)...")
    results['success_cmp'] = run_comparison(n_qubits=4, k=2, n_trials=3, shots=512)

    # 4. Query complexity
    print("\n[4/5] Query complexity validation...")
    results['query_complexity'] = measure_query_complexity()

    # 5. Circuit depth
    print("\n[5/5] Circuit depth comparison...")
    results['circuit_depth'] = measure_depths(
        n_qubits_range=[4, 5, 6] if quick else [4, 5, 6, 7, 8],
        k=4, trials=2, shots=512
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All experiments complete in {elapsed:.1f}s")
    print(f"Results saved to results/tables/")
    print(f"Run: python results/plot_results.py  to generate figures")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Fast subset for testing')
    args = parser.parse_args()
    run_all(quick=args.quick)
