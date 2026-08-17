
"""
Dataset Generator - Collects ineffective fault ciphertexts
SPM CS6630 - Final Project
Roll Number : CS25M048
"""

import os
import numpy as np
from aes_core import key_expansion, aes_with_fault

# ---- config ----

KEY = [0x2B,0x7E,0x15,0x16,0x28,0xAE,0xD2,0xA6,
       0xAB,0xF7,0x15,0x88,0x09,0xCF,0x4F,0x3C]

NUM_ATTEMPTS = 50000
FAULT_BIT    = 0
OUT_DIR      = "dataset"

# ---- collection ----

def collect_traces(key, fbyte, fbit, num_attempts, seed):
    rng = np.random.RandomState(seed)
    rks = key_expansion(key)

    good_cts = []
    good_pts = []
    n_eff = 0
    n_ineff = 0

    for i in range(num_attempts):
        pt = list(rng.randint(0, 256, 16))
        _, clean_ct, was_ineff = aes_with_fault(pt, rks, fbyte, fbit)

        if was_ineff:
            n_ineff += 1
            good_cts.append(clean_ct)
            good_pts.append(pt)
        else:
            n_eff += 1

        if (i+1) % 10000 == 0:
            print(f"    {i+1}/{num_attempts}: ineff={n_ineff} eff={n_eff}")

    return np.array(good_cts, dtype=np.uint8), np.array(good_pts, dtype=np.uint8)


# ---- main ----

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rks = key_expansion(KEY)

    print("Generating SIFA dataset (Round 8 stuck-at-0 fault)")
    print(f"Key:  {' '.join(f'{b:02X}' for b in KEY)}")
    print(f"K10:  {' '.join(f'{b:02X}' for b in rks[10])}")
    print(f"Attempts per byte: {NUM_ATTEMPTS}")
    print()

    save_data = {
        'key': np.array(KEY, dtype=np.uint8),
        'last_round_key': np.array(rks[10], dtype=np.uint8),
        'all_round_keys': np.array(rks, dtype=np.uint8),
    }

    for b in range(16):
        print(f"Byte {b:2d}:")
        cts, pts = collect_traces(KEY, b, FAULT_BIT, NUM_ATTEMPTS, seed=42+b)
        rate = len(cts) / NUM_ATTEMPTS * 100
        print(f"  -> {len(cts)} ineffective traces ({rate:.1f}%)\n")
        save_data[f'X_{b}'] = cts
        save_data[f'P_{b}'] = pts

    out_path = os.path.join(OUT_DIR, "sifa_dataset.npz")
    np.savez(out_path, **save_data)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
