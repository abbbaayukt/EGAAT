# EGAAT — Enhanced Grover's Algorithm with Adaptive Techniques

**Multi-Target Quantum Database Search | Qiskit | NISQ-Compatible**

---

## What is EGAAT?

EGAAT solves a fundamental limitation of Grover's algorithm: the original
algorithm requires knowing *k* (the number of targets) in advance.
In real applications, k is unknown. EGAAT fixes this with a 3-layer framework:

| Layer | Component | Innovation |
|-------|-----------|------------|
| 1 | Multi-target oracle | Marks arbitrary targets, depth O(log k) |
| 2 | QAE pre-processing | Estimates k without classical enumeration |
| 3 | Adaptive diffusion | Modulates strength by target density ρ=k/N |

**Results:** >94% success probability, O(√(N/k)) queries, stable up to 0.1% NISQ noise.

---

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.egaat.egaat_circuit import EGAATCircuit

egaat = EGAATCircuit(n_qubits=6, targets=[5, 12, 47, 55])
result = egaat.run(verbose=True)

print(f"Found: {result.found_targets[:3]}")
print(f"P(success): {result.success_probability_measured:.4f}")
```

## Run All Experiments

```bash
python experiments/run_all_experiments.py
python results/plot_results.py
```

## Run Tests

```bash
pytest src/oracle/oracle_tests.py -v
```

## Project Structure

```
EGAAT/
├── src/
│   ├── oracle/          # Multi-target phase oracle (Contribution 1)
│   ├── iteration_control/ # QAE estimator + dynamic iterations (Contribution 2)
│   ├── diffusion/       # Adaptive diffusion operator (Contribution 3)
│   └── egaat/           # Unified pipeline
├── baselines/           # Standard Grover, Fixed-Point, Quantum Walk
├── experiments/         # Benchmarks + noise sweep
├── results/             # Figures and tables
├── paper/               # LaTeX manuscript
└── applications/        # Real-world use cases
```

