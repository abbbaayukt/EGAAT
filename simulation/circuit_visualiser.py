"""
circuit_visualiser.py
=====================
Render and export EGAAT quantum circuits for paper figures.
Exports to PDF/PNG for inclusion in LaTeX manuscript.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use('Agg')

from src.oracle.multi_target_oracle import build_multi_target_oracle
from src.diffusion.adaptive_diffusion import adaptive_diffusion, compute_alpha
from src.egaat.egaat_circuit import EGAATCircuit


def draw_oracle(n=3, targets=None, save_path=None):
    if targets is None:
        targets = [2, 5]
    qc = build_multi_target_oracle(n, targets)
    fig = qc.draw(output='mpl', style='iqp')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def draw_full_egaat_circuit(n=3, targets=None, save_path=None):
    if targets is None:
        targets = [2, 5]
    egaat = EGAATCircuit(n, targets, shots=512)
    alpha = compute_alpha(len(targets), 2**n)
    from src.iteration_control.dynamic_iteration import optimal_iterations
    m = optimal_iterations(2**n, len(targets))
    qc = egaat.build_search_circuit(m, alpha)
    fig = qc.draw(output='mpl', style='iqp', fold=40)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


if __name__ == "__main__":
    draw_oracle(n=3, targets=[2,5], save_path='results/figures/oracle_circuit.pdf')
    draw_full_egaat_circuit(n=3, targets=[2,5], save_path='results/figures/egaat_full_circuit.pdf')
    print("Circuit figures exported.")
