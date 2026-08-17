
"""
Compare Traditional SIFA vs DL-SIFA
SPM CS6630 - Final Project
Roll Number : CS25M048
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aes_core import key_expansion
from traditional_sifa import (check_bias, recover_k10_byte as trad_recover,
                                progressive_recovery as trad_progressive)
from dl_sifa_model import (dl_sifa_attack, recover_k10_byte as nn_recover,
                            progressive_recovery as nn_progressive)


def main():
    os.makedirs("results", exist_ok=True)

    data = np.load(os.path.join("dataset", "sifa_dataset.npz"))
    key = list(data['key'])
    rks = [list(data['all_round_keys'][r]) for r in range(11)]
    true_byte = rks[10][0]

    print("=" * 60)
    print("Traditional SIFA vs DL-SIFA")
    print("=" * 60)
    print(f"Key: {' '.join(f'{b:02X}' for b in key)}")
    print(f"Target: K10[0] = 0x{true_byte:02X}")
    print()

    X0 = data['X_0']

    # ---- traditional SIFA ----
    print("--- Traditional SIFA ---")
    biases = []
    for b in range(16):
        X = data[f'X_{b}']
        bias = check_bias(X, b, fbit=0, rks=rks)
        biases.append(bias)
        ok = "PASS" if bias == 1.0 else "FAIL"
        print(f"  Byte {b:2d}: {bias:.4f} [{ok}]")

    print("\n  Key recovery (byte 0):")
    trad_best, trad_biases = trad_recover(X0, 0, 0, rks)
    print(f"  Recovered = 0x{trad_best:02X} ({'OK' if trad_best == true_byte else 'FAIL'})")

    steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    print(f"\n  Progressive:")
    trad_results = trad_progressive(X0, 0, 0, rks, steps)
    for r in trad_results:
        ok = "OK" if r['success'] else "FAIL"
        print(f"    n={r['n']:5d}: bias={r['correct_bias']:.4f} [{ok}]")

    # ---- DL-SIFA ----
    print("\n--- DL-SIFA ---")
    nn_best, nn_scores, model = dl_sifa_attack(X0, fbyte=0, fbit=0, rks=rks)

    print(f"\n  Progressive:")
    nn_results = nn_progressive(X0, 0, rks, model, steps)
    for r in nn_results:
        ok = "OK" if r['success'] else "FAIL"
        print(f"    n={r['n']:5d}: score={r['correct_score']:.4f} [{ok}]")

    # find minimum traces
    trad_min = next((r['n'] for r in trad_results if r['success']), None)
    nn_min   = next((r['n'] for r in nn_results if r['success']), None)

    print(f"\n--- Min traces for key recovery ---")
    print(f"  Traditional: {trad_min or '>5000'}")
    print(f"  DL-SIFA:     {nn_min or '>5000'}")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) bias per byte
    ax = axes[0][0]
    ax.bar(range(16), biases, color='steelblue')
    ax.axhline(1.0, color='red', ls='--', alpha=0.6)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Fault byte')
    ax.set_ylabel('Bit-0 bias')
    ax.set_title('Traditional SIFA: bias per byte')
    ax.set_ylim(0, 1.1)

    # 2) trad key distinguisher
    ax = axes[0][1]
    ax.bar(range(256), trad_biases, color='gray', alpha=0.5, width=1.0)
    ax.bar([true_byte], [trad_biases[true_byte]], color='green', width=3.0)
    ax.set_xlabel('K10[0] guess')
    ax.set_ylabel('Bias')
    ax.set_title(f'Traditional: 256 guesses (correct=0x{true_byte:02X})')
    ax.legend(['Wrong', f'Correct (0x{true_byte:02X})'], fontsize=9)

    # 3) DL-SIFA key distinguisher
    ax = axes[1][0]
    ax.bar(range(256), nn_scores, color='gray', alpha=0.5, width=1.0)
    ax.bar([true_byte], [nn_scores[true_byte]], color='coral', width=3.0)
    ax.set_xlabel('K10[0] guess')
    ax.set_ylabel('NN score')
    ax.set_title(f'DL-SIFA: 256 guesses (correct=0x{true_byte:02X})')
    ax.legend(['Wrong', f'Correct (0x{true_byte:02X})'], fontsize=9)

    # 4) progressive comparison
    ax = axes[1][1]
    t_steps = [r['n'] for r in trad_results]
    t_ok = [1 if r['success'] else 0 for r in trad_results]
    n_steps = [r['n'] for r in nn_results]
    n_ok = [1 if r['success'] else 0 for r in nn_results]
    ax.plot(t_steps, t_ok, 'g-o', label='Traditional SIFA', markersize=7)
    ax.plot(n_steps, n_ok, 'r-s', label='DL-SIFA (NN)', markersize=7)
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('Key recovered')
    ax.set_title('Traces needed for key recovery')
    ax.set_xscale('log')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['FAIL', 'SUCCESS'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=150)
    print(f"\nPlots saved to results/comparison.png")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Traditional: K10[0]=0x{trad_best:02X}, min traces = {trad_min or '>5000'}")
    print(f"  DL-SIFA:     K10[0]=0x{nn_best:02X}, min traces = {nn_min or '>5000'}")


if __name__ == "__main__":
    main()
