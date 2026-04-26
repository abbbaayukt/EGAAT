"""
quantum_id_matcher.py
=====================
Real-world application demo: Searching a database for matching IDs.
Simulates finding specific "Active" user IDs in a database of 64 entries.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.egaat.egaat_circuit import EGAATCircuit

def run_app_demo():
    print("=== EGAAT Application: Quantum ID Matcher ===")
    print("Database: 64 Users (6 qubits)")
    print("Target: Find all users with ID matching 'Active' flags [15, 33, 47]\n")

    # Define the "Active" IDs we are looking for
    active_ids = [15, 33, 47]
    
    # Initialize the EGAAT engine
    # In a real app, the oracle would be built from the database constraints
    matcher = EGAATCircuit(n_qubits=6, targets=active_ids, shots=2048)
    
    print("[1/3] Estimating number of active users...")
    result = matcher.run(verbose=True)
    
    print("\n[2/3] Decoding Quantum Results...")
    # Map the top results to "User Records"
    top_outcomes = result.found_targets[:5]
    
    print("\n[3/3] Final Report:")
    print("-" * 30)
    for rank, uid in enumerate(top_outcomes):
        status = "MATCH FOUND" if uid in active_ids else "Noise/Other"
        print(f"Rank {rank+1}: UserID {uid:>2} -> {status}")
    
    print("-" * 30)
    print(f"Success Probability: {result.success_probability_measured*100:.2f}%")
    print(f"Quantum Speedup: Found in {result.iterations_used} iterations (vs 32 classical checks)")

if __name__ == "__main__":
    run_app_demo()
