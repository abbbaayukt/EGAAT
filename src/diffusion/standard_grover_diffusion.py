"""
standard_grover_diffusion.py
Re-export of grover_diffusion from adaptive_diffusion.
"""
from src.diffusion.adaptive_diffusion import grover_diffusion, diffusion_matrix
from qiskit import QuantumCircuit
import numpy as np

__all__ = ['grover_diffusion', 'diffusion_matrix']

if __name__ == "__main__":
    for n in [2, 3, 4]:
        qc = grover_diffusion(n)
        print(f"n={n}: depth={qc.depth()}, gates={qc.size()}")
        print(qc.draw(output='text'))
        print()
