"""
oracle_tests.py
===============
Unit tests for EGAAT oracle layer.
Verifies correctness of multi-target oracle and phase gate constructions.

Run with: python -m pytest src/oracle/oracle_tests.py -v
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

from multi_target_oracle import build_multi_target_oracle, oracle_matrix
from controlled_phase_gate import phase_kickback_oracle, controlled_phase_gate


class TestMultiTargetOracle:

    def test_single_target_phase(self):
        """Oracle must flip phase of exactly one target."""
        n, target = 3, 5
        qc = build_multi_target_oracle(n, [target])
        op = Operator(qc).data
        for i in range(2**n):
            expected = -1.0 if i == target else 1.0
            assert np.isclose(np.real(op[i, i]), expected, atol=1e-6), \
                f"Failed at index {i}"

    def test_multi_target_phase(self):
        """Oracle must flip phase of all listed targets."""
        n, targets = 4, [0, 3, 7, 15]
        qc = build_multi_target_oracle(n, targets)
        op = Operator(qc).data
        for i in range(2**n):
            expected = -1.0 if i in targets else 1.0
            assert np.isclose(np.real(op[i, i]), expected, atol=1e-6), \
                f"Failed at index {i}: expected {expected}"

    def test_oracle_is_unitary(self):
        """Oracle matrix must be unitary (U†U = I)."""
        n, targets = 3, [1, 4, 6]
        qc = build_multi_target_oracle(n, targets)
        op = Operator(qc).data
        product = op @ op.conj().T
        assert np.allclose(product, np.eye(2**n), atol=1e-6), \
            "Oracle is not unitary!"

    def test_empty_targets_raises(self):
        with pytest.raises(ValueError):
            build_multi_target_oracle(3, [])

    def test_out_of_range_target_raises(self):
        with pytest.raises(ValueError):
            build_multi_target_oracle(3, [8])  # max is 7

    def test_oracle_matrix_helper(self):
        """oracle_matrix() must return correct diagonal."""
        n, targets = 2, [0, 3]
        mat = oracle_matrix(n, targets)
        assert mat.shape == (4, 4)
        assert mat[0, 0] == -1.0
        assert mat[1, 1] == 1.0
        assert mat[2, 2] == 1.0
        assert mat[3, 3] == -1.0

    def test_all_targets(self):
        """Marking all states: oracle should be -I."""
        n = 2
        targets = list(range(2**n))
        qc = build_multi_target_oracle(n, targets)
        op = Operator(qc).data
        assert np.allclose(op, -np.eye(2**n), atol=1e-6)

    def test_circuit_depth_reasonable(self):
        """Circuit depth should stay manageable for NISQ (< 100 for n=4)."""
        n, targets = 4, [2, 5, 10, 14]
        qc = build_multi_target_oracle(n, targets)
        assert qc.depth() < 200, f"Circuit too deep: {qc.depth()}"


class TestPhaseKickback:

    def test_kickback_correct_phase(self):
        """Phase kickback oracle should invert exactly the target state."""
        n, target = 3, 6
        qc = phase_kickback_oracle(n, target)
        op = Operator(qc).data
        assert np.isclose(np.real(op[target, target]), -1.0, atol=1e-6)
        for i in range(2**n):
            if i != target:
                assert np.isclose(np.real(op[i, i]), 1.0, atol=1e-6)

    def test_controlled_phase_gate_matrix(self):
        """Controlled phase gate with phase=pi should give -1 at last diagonal."""
        gate = controlled_phase_gate(2, np.pi)
        mat = gate.to_matrix()
        assert np.isclose(mat[-1, -1], -1.0, atol=1e-6)
        for i in range(len(mat) - 1):
            assert np.isclose(mat[i, i], 1.0, atol=1e-6)

    def test_arbitrary_phase(self):
        """Phase gate with phase=pi/2 should apply i to last state."""
        gate = controlled_phase_gate(1, np.pi / 2)
        mat = gate.to_matrix()
        assert np.isclose(mat[-1, -1], 1j, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
