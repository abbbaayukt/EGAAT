"""
genomic_pattern_match.py
========================
Application: EGAAT for Genomic Pattern Matching

Models searching a database of encoded DNA k-mers for sequences
matching a given motif. Multiple matches are expected (multi-target).

Each database entry is a binary-encoded DNA fragment.
The oracle marks entries matching the target motif.

Reference: EGAAT §1.4 (motivation — bioinformatics application)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from typing import List


# DNA encoding: A=00, T=01, G=10, C=11
DNA_MAP = {'A': 0b00, 'T': 0b01, 'G': 0b10, 'C': 0b11}
REV_DNA = {v: k for k, v in DNA_MAP.items()}


def encode_kmer(kmer: str) -> int:
    """Encode a DNA k-mer as a binary integer."""
    result = 0
    for base in kmer:
        result = (result << 2) | DNA_MAP[base.upper()]
    return result


def decode_kmer(code: int, k: int) -> str:
    """Decode a binary integer back to a DNA k-mer."""
    bases = []
    for _ in range(k):
        bases.append(REV_DNA[code & 0b11])
        code >>= 2
    return ''.join(reversed(bases))


def find_motif_targets(db_size_bits: int, motif: str, wildcard_pos: List[int] = None) -> List[int]:
    """
    Find all database entries (encoded k-mers) that match a motif.
    Wildcard positions match any base.

    Parameters
    ----------
    db_size_bits : int
        n_qubits — database has 2^n entries.
    motif : str
        DNA sequence, e.g. 'ATGC'. Length = db_size_bits // 2.
    wildcard_pos : List[int]
        Positions (0-indexed) that are wildcards (match any base).
    """
    kmer_len = db_size_bits // 2
    N = 2 ** db_size_bits
    if wildcard_pos is None:
        wildcard_pos = []

    motif_code = encode_kmer(motif[:kmer_len].ljust(kmer_len, 'A'))
    targets = []

    for i in range(N):
        kmer = decode_kmer(i, kmer_len)
        match = True
        for pos in range(min(len(motif), kmer_len)):
            if pos in wildcard_pos:
                continue
            if pos >= len(motif) or kmer[pos] != motif[pos].upper():
                match = False
                break
        if match:
            targets.append(i)

    return targets


def run_genomic_search(n_qubits: int = 6, motif: str = 'ATG', wildcard_pos=None):
    """
    Run EGAAT for genomic motif search.
    Demonstrates multi-target nature: many k-mers may match a motif.
    """
    from src.egaat.egaat_circuit import EGAATCircuit

    if wildcard_pos is None:
        wildcard_pos = [2]  # Position 2 is wildcard (degenerate base)

    targets = find_motif_targets(n_qubits, motif, wildcard_pos)
    N = 2 ** n_qubits

    print(f"Genomic Search: motif='{motif}' (pos {wildcard_pos} = wildcard)")
    print(f"  Database: {N} encoded {n_qubits//2}-mers")
    print(f"  Matching entries (targets): {len(targets)} ({len(targets)/N*100:.1f}%)")

    if not targets:
        print("  No matches found — adjust motif or wildcards.")
        return None

    egaat = EGAATCircuit(n_qubits, targets, shots=2048)
    result = egaat.run(verbose=True)

    # Decode found results back to DNA
    kmer_len = n_qubits // 2
    found_kmers = [decode_kmer(idx, kmer_len) for idx in result.found_targets[:5]]

    print(f"\nFound k-mers: {found_kmers}")
    print(f"P(success): {result.success_probability_measured:.4f}")
    return result


if __name__ == "__main__":
    print("=== EGAAT: Genomic Pattern Matching ===\n")

    # Encode ATG with wildcard at position 2
    run_genomic_search(n_qubits=6, motif='ATG', wildcard_pos=[2])

    print("\n--- Encoding examples ---")
    for kmer in ['ATGC', 'GAAC', 'TTTT']:
        code = encode_kmer(kmer)
        back = decode_kmer(code, len(kmer))
        print(f"  {kmer} -> {code:08b} ({code}) -> {back}")
