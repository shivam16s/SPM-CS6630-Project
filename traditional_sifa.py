
"""
Traditional SIFA - Key Recovery using bit-bias
SPM CS6630 - Final Project
Roll Number : CS25M048
"""

import os
import numpy as np
from aes_core import (key_expansion, SBOX, INV_SBOX, add_round_key,
                       inv_shift_rows, inv_sub_bytes, inv_mix_columns)


# ---- inversion ----

def invert_to_r8(ct, rks):
    # undo round 10, 9, 8 to reach R8 SubBytes input
    s = list(ct)
    s = add_round_key(s, rks[10])
    s = inv_shift_rows(s)
    s = inv_sub_bytes(s)
    s = add_round_key(s, rks[9])
    s = inv_mix_columns(s)
    s = inv_shift_rows(s)
    s = inv_sub_bytes(s)
    s = add_round_key(s, rks[8])
    s = inv_mix_columns(s)
    s = inv_shift_rows(s)
    s = inv_sub_bytes(s)
    return s


def check_bias(cts, fbyte, fbit, rks):
    # check what fraction of traces have bit=0 at fault position
    mask = 1 << fbit
    zeros = sum(1 for ct in cts
                if (invert_to_r8(list(ct), rks)[fbyte] & mask) == 0)
    return zeros / len(cts)


# ---- key recovery ----

def recover_k10_byte(cts, fbyte, fbit, rks):
    # try all 256 guesses for K10[fbyte]
    # correct key -> bias = 1.0, wrong keys -> bias ~0.5
    biases = np.zeros(256)
    true_k10 = list(rks[10])

    for guess in range(256):
        test_rks = [list(rk) for rk in rks]
        test_rks[10] = list(true_k10)
        test_rks[10][fbyte] = guess
        biases[guess] = check_bias(cts, fbyte, fbit, test_rks)

    best = int(np.argmax(biases))
    return best, biases


def progressive_recovery(cts, fbyte, fbit, rks, steps):
    # try recovery with increasing number of traces
    true_byte = rks[10][fbyte]
    results = []
    for n in steps:
        if n > len(cts):
            break
        best, biases = recover_k10_byte(cts[:n], fbyte, fbit, rks)
        results.append({
            'n': n,
            'success': (best == true_byte),
            'correct_bias': biases[true_byte],
            'best_guess': best,
            'avg_wrong': np.mean([biases[g] for g in range(256) if g != true_byte]),
        })
    return results


# ---- main ----

def main():
    data = np.load(os.path.join("dataset", "sifa_dataset.npz"))
    key = list(data['key'])
    rks = [list(data['all_round_keys'][r]) for r in range(11)]

    print("Traditional SIFA - Key Recovery")
    print(f"Key:  {' '.join(f'{b:02X}' for b in key)}")
    print(f"K10:  {' '.join(f'{b:02X}' for b in rks[10])}")
    print()

    # check bias on all 16 bytes
    print("--- Bias check ---")
    for b in range(16):
        X = data[f'X_{b}']
        bias = check_bias(X, b, fbit=0, rks=rks)
        ok = "PASS" if bias == 1.0 else "FAIL"
        print(f"  Byte {b:2d}: bias={bias:.4f} ({int(bias*len(X))}/{len(X)}) [{ok}]")

    # key recovery for byte 0
    print("\n--- Key recovery (byte 0) ---")
    X0 = data['X_0']
    true_byte = rks[10][0]
    best, biases = recover_k10_byte(X0, 0, 0, rks)
    print(f"  True K10[0] = 0x{true_byte:02X}")
    print(f"  Recovered   = 0x{best:02X}")
    print(f"  Correct bias: {biases[true_byte]:.4f}")
    print(f"  Avg wrong:    {np.mean([biases[g] for g in range(256) if g != true_byte]):.4f}")
    print(f"  {'SUCCESS' if best == true_byte else 'FAIL'}")

    # progressive
    print("\n--- Progressive ---")
    steps = [10, 20, 50, 100, 200, 500, 1000]
    results = progressive_recovery(X0, 0, 0, rks, steps)
    for r in results:
        ok = "OK" if r['success'] else "FAIL"
        print(f"  n={r['n']:5d}: bias={r['correct_bias']:.4f} "
              f"avg_wrong={r['avg_wrong']:.4f} [{ok}]")


if __name__ == "__main__":
    main()
