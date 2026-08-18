# SPM CS6630 — Statistical Fault Attack on AES-128 (SIFA & DL-SIFA)

> **Course:** CS6630 – Statistical Methods in Security (SPM)
> **Author:** CS25M048

---

## Project Overview

This project implements and compares two approaches to performing a **Statistical Ineffective Fault Attack (SIFA)** on an AES-128 encryption engine to recover its secret key:

1. **Traditional SIFA** — A classical statistical approach using bit-bias analysis.
2. **DL-SIFA** — A modern deep learning-based approach using a neural network trained to detect histogram bias.

The attack exploits *ineffective faults* — fault injections that, due to the deterministic nature of the fault model (stuck-at-0), do not change the ciphertext output. These ineffective traces carry a statistical bias in the intermediate values of the cipher, which can be used to uniquely identify the correct key byte.

---

## Repository Structure

```
.
├── aes_core.py           # Full AES-128 implementation from scratch + fault injection
├── generate_dataset.py   # Dataset generator (collects ineffective fault traces)
├── traditional_sifa.py   # Traditional SIFA: bit-bias key recovery
├── dl_sifa_model.py      # DL-SIFA: neural network key recovery
├── compare_results.py    # Comparison script + plots
├── run_all.py            # One-click full pipeline runner
├── dataset/
│   └── sifa_dataset.npz  # Pre-generated fault traces (16 bytes × ~50000 attempts)
├── results/
│   ├── comparison.png         # Main comparison plot
│   ├── comparison_plots.png   # Detailed plots
│   ├── dl_progressive.png     # DL-SIFA progressive recovery curve
│   └── dl_sifa_progressive.png
└── README.md
```

---

## File-by-File Explanation

### `aes_core.py` — AES-128 from Scratch
This is the cryptographic foundation of the project, implementing the full **AES-128 (FIPS-197)** standard entirely in Python.

**Key components:**
- **`SBOX` / `INV_SBOX`**: The standard 256-entry AES substitution box and its inverse, defined as lookup tables.
- **`RCON`**: Round constant array used during key schedule expansion.
- **`gf_mult(a, b)`**: Galois Field GF(2⁸) multiplication (polynomial multiplication modulo the AES irreducible polynomial `0x11B`). Uses a bit-by-bit Russian Peasant algorithm with conditional XOR with `0x1B` (the field reduction constant). Pre-computes lookup tables for multipliers `{2, 3, 9, 11, 13, 14}`.
- **`sub_bytes(s)` / `inv_sub_bytes(s)`**: Applies the S-BOX substitution to each byte of the state (list of 16 bytes).
- **`shift_rows(s)` / `inv_shift_rows(s)`**: Cyclically shifts rows of the 4×4 AES state matrix. Rows 0, 1, 2, 3 are shifted left by 0, 1, 2, 3 positions respectively (inverse shifts right).
- **`mix_columns(s)` / `inv_mix_columns(s)`**: Multiplies each column of the state by the MixColumns matrix over GF(2⁸). Uses the precomputed GF multiplication tables for efficiency.
- **`add_round_key(s, rk)`**: XORs the state with the current round key.
- **`key_expansion(key)`**: Implements the AES-128 key schedule, generating all 11 round keys from the initial 16-byte key. Returns a list of 11 round keys.
- **`aes_encrypt(pt, rks)`**: Performs the full 10-round AES-128 encryption: 1 initial AddRoundKey + 9 full rounds (SubBytes → ShiftRows → MixColumns → AddRoundKey) + 1 final round (no MixColumns).
- **`aes_with_fault(pt, rks, fbyte, fbit)`**: **The core of SIFA**. Runs the AES encryption but injects a **stuck-at-0 fault** at the *input to Round 8*, specifically at bit `fbit` of byte `fbyte`. Returns the faulted ciphertext, the clean ciphertext, and a flag indicating whether the fault was **ineffective** (faulted byte == clean byte, meaning the bit was already 0).
- **`protected_encrypt(pt, rks, fbyte, fbit)`**: Simulates a **temporal redundancy** countermeasure. Encrypts twice and compares outputs; if they differ the fault is suppressed.

---

### `generate_dataset.py` — Fault Trace Collection
Generates the dataset of ineffective fault ciphertexts used for both attacks.

**Algorithm:**
1. Uses the fixed NIST AES-128 test key.
2. For each of the **16 bytes** of the AES state at Round 8, attempts `50,000` fault injections with `FAULT_BIT=0` (stuck-at-0 on bit 0).
3. For each attempt, generates a random plaintext, calls `aes_with_fault`, and records the **clean ciphertext only if the fault was ineffective**.
4. Saves all results to `dataset/sifa_dataset.npz` containing:
   - `key`: The 16-byte secret key.
   - `last_round_key`: The 16-byte Round 10 key (the target of the attack).
   - `all_round_keys`: All 11 round keys (shape `11×16`).
   - `X_0` ... `X_15`: Arrays of ineffective ciphertexts for each fault byte.
   - `P_0` ... `P_15`: Corresponding plaintext arrays.

**Expected ineffectiveness rate:** ~50% (since each bit is 0 roughly half the time), so ~25,000 clean traces are collected per byte.

---

### `traditional_sifa.py` — Classical Bit-Bias Key Recovery
Implements the statistical SIFA attack using raw bit-bias analysis.

**Theory:** When the fault is *stuck-at-0* on bit `j` of byte `b` at Round 8 input, and the fault is *ineffective*, it means that bit was already 0 before the fault. This causes a 100% bias: all collected traces have bit `j = 0` at that position. For a **wrong** key guess, the inversion path scrambles values and the bias drops to ~50%.

**Key functions:**
- **`invert_to_r8(ct, rks)`**: Inverts a ciphertext back through rounds 10, 9, 8 using `InvShiftRows`, `InvSubBytes`, `InvMixColumns`, and `AddRoundKey` to reach the input of Round 8's SubBytes operation.
- **`check_bias(cts, fbyte, fbit, rks)`**: Given a list of ciphertexts and key guesses, inverts each to Round 8 input and calculates the fraction where the target bit is 0.
- **`recover_k10_byte(cts, fbyte, fbit, rks)`**: Iterates all 256 possible values for `K10[fbyte]` (the last round key byte). For each guess, uses `check_bias` and returns the guess with the highest bias (ideally 1.0 for the correct key).
- **`progressive_recovery(cts, fbyte, fbit, rks, steps)`**: Evaluates key recovery success at increasing numbers of traces to characterize the minimum required.

---

### `dl_sifa_model.py` — Deep Learning Key Recovery
Replaces the explicit bias threshold with a trained **neural network classifier**.

**Architecture (`BiasNet`):**
```
Input: 256-bin normalized histogram of byte values at R8 input
→ Linear(256→128) + BatchNorm + ReLU + Dropout(0.3)
→ Linear(128→64)  + BatchNorm + ReLU + Dropout(0.2)
→ Linear(64→32)   + ReLU
→ Linear(32→1)    + Sigmoid
Output: Score in [0,1] → 1 = biased (correct key), 0 = unbiased
```

**Training Pipeline (`build_training_data`):**
- Randomly generates 500 different AES keys.
- For each key, collects `~500` ineffective fault traces to build a **biased histogram**.
- Also collects clean traces to build an **unbiased histogram**.
- Creates histograms at multiple sample sizes (100, 200, 500) for robustness.
- Trains `BiasNet` using **Binary Cross-Entropy (BCE) loss** and **Adam optimizer** with L2 regularization for 200 epochs.

**Key Recovery (`recover_k10_byte`):**
- For all 256 key guesses, inverts the attack ciphertexts to Round 8 input, builds a histogram of byte values, and feeds it to the trained network.
- The guess that scores closest to 1.0 is the recovered key byte.

---

### `compare_results.py` — Benchmarking & Visualization
Runs both attacks on the same dataset and produces comparison plots.

**Plots generated (`results/comparison.png`):**
1. **Bias per byte** (bar chart): Shows that traditional SIFA achieves perfect bias (1.0) for all 16 fault positions.
2. **Traditional SIFA key distinguisher** (bar chart): Shows the 256-guess bias landscape — the correct key byte stands out with bias=1.0.
3. **DL-SIFA key distinguisher** (bar chart): Shows the NN score landscape — the correct key byte is scored near 1.0.
4. **Progressive comparison** (line plot, log scale): Shows how many traces each method needs to successfully recover the key, enabling a direct efficiency comparison.

---

### `run_all.py` — Full Pipeline Orchestration
A top-level script that sequentially executes the entire pipeline:
1. **AES Verification** — Checks against the NIST test vector.
2. **Dataset Generation** — Runs `generate_dataset.py` if the dataset doesn't exist yet.
3. **Traditional SIFA** — Runs the classical attack.
4. **DL-SIFA** — Trains the neural network and runs the learned attack.
5. **Comparison** — Generates all plots and prints the summary.

---

## How to Run

### Prerequisites
```bash
pip install numpy torch matplotlib
```

### Full Pipeline (Recommended)
```bash
python run_all.py
```

### Individual Steps
```bash
# Step 1: Verify AES implementation
python aes_core.py

# Step 2: Generate the fault dataset
python generate_dataset.py

# Step 3: Traditional SIFA key recovery
python traditional_sifa.py

# Step 4: DL-SIFA key recovery
python dl_sifa_model.py

# Step 5: Compare both methods and generate plots
python compare_results.py
```

---

## Key Concepts

| Concept | Description |
|---|---|
| **SIFA** | Statistical Ineffective Fault Attack — exploits faults that do not change the output |
| **Stuck-at-0** | Fault model where a bit is forced to 0 regardless of its logical value |
| **Ineffective fault** | A fault injection where the output is identical to the unfaulted output |
| **Bit-bias** | The statistical property that the target bit is always 0 in ineffective traces |
| **Temporal Redundancy** | Countermeasure: encrypt twice and compare to detect faults |
| **DL-SIFA** | Uses a neural network trained to distinguish biased from unbiased histograms |

---

## Results

Both methods successfully recover `K10[0]` of the target AES-128 key. The progressive analysis shows how many traces each approach requires. Results and plots are saved in the `results/` directory.

---

## References

- Dobraunig et al., "SIFA: Exploiting Ineffective Fault Inductions on Symmetric Cryptography" (IACR TCHES 2018) — `3338508.3359572.pdf`
