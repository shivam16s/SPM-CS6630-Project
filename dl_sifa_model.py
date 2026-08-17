
"""
DL-SIFA - Neural Network Key Recovery
SPM CS6630 - Final Project
Roll Number : CS25M048
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from aes_core import (key_expansion, aes_with_fault, aes_encrypt,
                       add_round_key, inv_shift_rows,
                       inv_sub_bytes, inv_mix_columns)


# ---- neural network ----
# input: 256-bin histogram of byte value at R8 fault point
# output: score between 0 and 1 (1 = biased / faulted)

class BiasNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ---- helpers ----

def invert_to_r8(ct, rks):
    # undo R10 -> R9 -> R8 to reach R8 SubBytes input
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


def make_histogram(values):
    h = np.zeros(256, dtype=np.float32)
    for v in values:
        h[int(v)] += 1
    if len(values) > 0:
        h /= len(values)
    return h


# ---- profiling ----
# for many random keys, collect biased (faulted) and clean histograms

def build_training_data(n_keys=500, traces_per_key=500,
                         fbyte=0, fbit=0, seed=200):
    rng = np.random.RandomState(seed)
    X_all, y_all = [], []

    for ki in range(n_keys):
        rand_key = list(rng.randint(0, 256, 16))
        rks = key_expansion(rand_key)

        # get ineffective-fault CTs
        ineff_cts = []
        attempts = 0
        while len(ineff_cts) < traces_per_key and attempts < traces_per_key*4:
            pt = list(rng.randint(0, 256, 16))
            _, cct, was_ineff = aes_with_fault(pt, rks, fbyte, fbit)
            if was_ineff:
                ineff_cts.append(cct)
            attempts += 1

        if len(ineff_cts) < 100:
            continue

        # get byte values at R8 input (faulted and clean)
        vals_f = [invert_to_r8(ct, rks)[fbyte] for ct in ineff_cts]
        vals_c = []
        for _ in range(len(ineff_cts)):
            pt = list(rng.randint(0, 256, 16))
            ct = aes_encrypt(pt, rks)
            vals_c.append(invert_to_r8(ct, rks)[fbyte])

        # make histogram features at different sizes
        for sz in [100, 200, len(ineff_cts)]:
            if sz > len(ineff_cts):
                continue
            idx = list(rng.choice(len(vals_f), sz, replace=False))
            X_all.append(make_histogram([vals_f[i] for i in idx]))
            y_all.append(1.0)  # biased

            idx2 = list(rng.choice(len(vals_c), sz, replace=False))
            X_all.append(make_histogram([vals_c[i] for i in idx2]))
            y_all.append(0.0)  # clean

        if (ki+1) % 100 == 0:
            print(f"    profiled {ki+1}/{n_keys} keys ({len(X_all)} samples)")

    return np.array(X_all, dtype=np.float32), \
           np.array(y_all, dtype=np.float32).reshape(-1,1)


def train_model(X_train, y_train, epochs=200):
    model = BiasNet()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=64, shuffle=True
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += ((out > 0.5).float() == by).sum().item()
            total += len(by)
        if (ep+1) % 50 == 0:
            print(f"    epoch {ep+1}: loss={total_loss/len(loader):.4f} "
                  f"acc={correct/total*100:.1f}%")
    return model


# ---- key recovery ----

def score_histogram(model, h):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(h).unsqueeze(0)).item()


def recover_k10_byte(cts, fbyte, rks, model):
    # try all 256 guesses for K10[fbyte]
    # correct key -> histogram shows only even values -> high NN score
    scores = np.zeros(256)
    true_k10 = list(rks[10])

    for guess in range(256):
        test_rks = [list(rk) for rk in rks]
        test_rks[10] = list(true_k10)
        test_rks[10][fbyte] = guess
        vals = [invert_to_r8(list(ct), test_rks)[fbyte] for ct in cts]
        scores[guess] = score_histogram(model, make_histogram(vals))

    best = int(np.argmax(scores))
    return best, scores


def progressive_recovery(cts, fbyte, rks, model, steps):
    true_byte = rks[10][fbyte]
    results = []
    for n in steps:
        if n > len(cts):
            break
        best, scores = recover_k10_byte(cts[:n], fbyte, rks, model)
        results.append({
            'n': n,
            'success': (best == true_byte),
            'correct_score': scores[true_byte],
            'best_guess': best,
            'avg_wrong': np.mean([scores[g] for g in range(256) if g != true_byte]),
        })
    return results


# ---- main ----

def dl_sifa_attack(cts, fbyte, fbit, rks, n_profile_keys=500,
                    traces_per_key=500, epochs=200):
    print(f"  Feature: 256-bin histogram of byte {fbyte} at R8 input")

    # profiling
    print("  building profiling data...")
    t0 = time.time()
    X_train, y_train = build_training_data(
        n_keys=n_profile_keys, traces_per_key=traces_per_key,
        fbyte=fbyte, fbit=fbit
    )
    n_pos = int(np.sum(y_train))
    print(f"  done in {time.time()-t0:.1f}s ({len(X_train)} samples: "
          f"{n_pos} biased, {len(X_train)-n_pos} clean)")

    # training
    print(f"  training ({epochs} epochs)...")
    model = train_model(X_train, y_train, epochs)

    # key recovery
    true_byte = rks[10][fbyte]
    print(f"\n  Recovering K10[{fbyte}] (true = 0x{true_byte:02X})...")
    best, scores = recover_k10_byte(cts, fbyte, rks, model)
    print(f"  Recovered = 0x{best:02X}")
    print(f"  Correct key score: {scores[true_byte]:.4f}")
    print(f"  Avg wrong score:   {np.mean([scores[g] for g in range(256) if g != true_byte]):.4f}")
    print(f"  {'SUCCESS' if best == true_byte else 'FAIL'}")

    return best, scores, model


def main():
    data = np.load(os.path.join("dataset", "sifa_dataset.npz"))
    key = list(data['key'])
    rks = [list(data['all_round_keys'][r]) for r in range(11)]

    print("DL-SIFA - Neural Network Key Recovery")
    print(f"Key: {' '.join(f'{b:02X}' for b in key)}")

    X0 = data['X_0']
    print(f"Traces: {len(X0)} (fault at byte 0)\n")

    best, scores, model = dl_sifa_attack(X0, fbyte=0, fbit=0, rks=rks)

    # try with fewer traces
    print("\n--- Progressive ---")
    steps = [10, 20, 50, 100, 200, 500, 1000]
    results = progressive_recovery(X0, 0, rks, model, steps)
    for r in results:
        ok = "OK" if r['success'] else "FAIL"
        print(f"  n={r['n']:5d}: score={r['correct_score']:.4f} "
              f"avg_wrong={r['avg_wrong']:.4f} [{ok}]")


if __name__ == "__main__":
    main()
