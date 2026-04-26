"""
qiskit_backend.py
=================
Centralised Qiskit backend factory for EGAAT simulations.

All modules import their backend from here so switching between
statevector, noisy, and real hardware is a single change.
"""

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


def get_statevector_backend() -> AerSimulator:
    """Ideal noiseless statevector simulation. Exact probabilities."""
    return AerSimulator(method='statevector')


def get_mps_backend() -> AerSimulator:
    """Matrix Product State backend — scales to more qubits."""
    return AerSimulator(method='matrix_product_state')


def get_noisy_backend(p1: float = 0.001, p2: float = None) -> AerSimulator:
    """Depolarising noise backend."""
    from experiments.noise_models.depolarising_noise import build_depolarising_model
    if p2 is None:
        p2 = p1 * 10
    nm = build_depolarising_model(p1=p1, p2=p2)
    return AerSimulator(noise_model=nm)


def get_backend(mode: str = 'statevector', **kwargs) -> AerSimulator:
    """
    Factory function.

    Parameters
    ----------
    mode : str
        'statevector' | 'mps' | 'noisy'
    **kwargs
        Passed to mode-specific constructor (e.g. p1=0.001 for noisy).
    """
    if mode == 'statevector':
        return get_statevector_backend()
    elif mode == 'mps':
        return get_mps_backend()
    elif mode == 'noisy':
        return get_noisy_backend(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    for mode in ['statevector', 'mps']:
        b = get_backend(mode)
        print(f"{mode}: {b}")
    b_noisy = get_backend('noisy', p1=0.001)
    print(f"noisy: {b_noisy}")
