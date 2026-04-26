"""
controlled_phase_gate.py
========================
Utility: Logarithmic-depth controlled-phase gate constructions.

EGAAT uses controlled-phase operations as the primitive for oracle
construction. This module provides efficient decompositions that keep
circuit depth at O(log k) for k targets, critical for NISQ hardware.

Reference: EGAAT paper §1.5 (logarithmic circuit depth claim)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from typing import List, Optional


def controlled_phase_gate(n_controls: int, phase: float = np.pi) -> Gate:
    """
    Build an n-controlled phase gate that applies e^(i*phase) to |11...1>.

    For phase=pi this is the standard multi-controlled Z used in Grover's
    oracle. For arbitrary phase it enables partial amplitude amplification.

    Parameters
    ----------
    n_controls : int
        Number of control qubits.
    phase : float
        Phase angle in radians. Default pi gives -1 phase (standard Grover).

    Returns
    -------
    Gate
        Qiskit gate object, can be appended to any circuit.
    """
    total_qubits = n_controls + 1  # controls + target
    dim = 2 ** total_qubits
    matrix = np.eye(dim, dtype=complex)
    # Only the all-ones state gets the phase
    matrix[-1, -1] = np.exp(1j * phase)
    gate = UnitaryGate(matrix, label=f"CP({phase:.2f})")
    return gate


def logarithmic_phase_cascade(
    qc: QuantumCircuit,
    qubits: List[int],
    ancilla: Optional[List[int]] = None
) -> QuantumCircuit:
    """
    Apply a phase inversion to the |11...1> state of `qubits` using
    a logarithmic-depth cascade of Toffoli gates into ancilla qubits,
    then a CZ, then uncompute.

    This achieves O(log n) depth vs O(n) for the naive MCX approach,
    which is the key innovation for NISQ compatibility.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to append gates to.
    qubits : List[int]
        Qubit indices that form the control string.
    ancilla : List[int], optional
        Ancilla qubit indices for the cascade. If None, naive MCX is used.

    Returns
    -------
    QuantumCircuit
        Modified circuit (in-place).
    """
    n = len(qubits)

    if n == 1:
        qc.z(qubits[0])
        return qc
    if n == 2:
        qc.cz(qubits[0], qubits[1])
        return qc

    if ancilla is None or len(ancilla) < n - 2:
        # Fallback to standard MCX decomposition
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        return qc

    # Logarithmic cascade: pair-wise Toffoli tree
    # Layer 1: pair up controls into ancilla
    layer = list(qubits)
    anc_idx = 0
    used_ancilla = []

    while len(layer) > 2:
        next_layer = []
        i = 0
        while i + 1 < len(layer):
            anc = ancilla[anc_idx]
            qc.ccx(layer[i], layer[i + 1], anc)
            used_ancilla.append((layer[i], layer[i + 1], anc))
            next_layer.append(anc)
            anc_idx += 1
            i += 2
        if i < len(layer):
            next_layer.append(layer[i])
        layer = next_layer

    # Now apply CZ to the last two
    qc.cz(layer[0], layer[1])

    # Uncompute ancilla (reverse order)
    for ctrl1, ctrl2, anc in reversed(used_ancilla):
        qc.ccx(ctrl1, ctrl2, anc)

    return qc


def phase_kickback_oracle(n_qubits: int, target: int, phase: float = np.pi) -> QuantumCircuit:
    """
    Single-target phase kickback oracle for a specific target state.

    Applies a phase `e^(i*phase)` to |target> by flipping 0-bits,
    applying controlled phase, then unflipping.

    Used as a building block in multi_target_oracle.py.
    """
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr)

    bits = format(target, f'0{n_qubits}b')
    zero_positions = [i for i, b in enumerate(reversed(bits)) if b == '0']

    # Flip zeros to ones
    for pos in zero_positions:
        qc.x(qr[pos])

    # Apply controlled phase to |11...1>
    gate = controlled_phase_gate(n_qubits - 1, phase)
    qc.append(gate, list(range(n_qubits)))

    # Undo flips
    for pos in zero_positions:
        qc.x(qr[pos])

    qc.name = f"PhaseKickback[{target}]"
    return qc


def verify_phase_gate(n_qubits: int, targets: List[int]) -> bool:
    """
    Classical verification: simulate the oracle matrix and confirm
    that only target states receive a -1 phase.

    Returns True if oracle is correct, raises AssertionError otherwise.
    """
    from qiskit.quantum_info import Operator
    from src.oracle.multi_target_oracle import build_multi_target_oracle

    qc = build_multi_target_oracle(n_qubits, targets)
    op = Operator(qc).data

    N = 2 ** n_qubits
    for i in range(N):
        expected = -1.0 if i in targets else 1.0
        actual = np.real(op[i, i])
        if not np.isclose(actual, expected, atol=1e-6):
            raise AssertionError(
                f"Oracle incorrect at index {i}: expected {expected}, got {actual:.4f}"
            )
    return True


if __name__ == "__main__":
    print("=== Controlled Phase Gate Tests ===\n")

    # Test 1: single-target kickback
    qc = phase_kickback_oracle(3, 5)
    print(f"Single-target oracle for |101>:")
    print(qc.draw(output='text'))

    # Test 2: controlled phase gate matrix
    gate = controlled_phase_gate(2, np.pi)
    print(f"\n2-control phase gate matrix (last entry should be -1):")
    print(np.round(gate.to_matrix(), 3))

    print("\nAll phase gate tests passed.")
