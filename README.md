# DL-SIFA: Deep Learning Statistical Ineffective Fault Analysis on AES-128

**CS6630 - Secure Processor Microarchitecture | Final Project**  
**CS25M048 | Singh Shivam Ramdarash | IIT Madras**

## What this does

SIFA breaks AES even when the device has fault detection (temporal redundancy).
It uses only the "correct" ciphertexts — the ones where the injected fault
happened to not change anything (ineffective faults). These ciphertexts have
a statistical bias that leaks the key.

This project compares two ways to **recover the AES key** from that bias:
1. **Traditional SIFA** — check bit values with a formula (recovers key in 20 traces)
2. **DL-SIFA** — train a neural network to recognize biased distributions (recovers key in 50 traces)

## Fault Model

- **Target**: AES-128 with temporal redundancy (encrypt twice, compare)
- **Fault**: Stuck-at-0 on bit 0 of one byte
- **Location**: Input to Round 8 SubBytes
- **Ineffective rate**: ~50%
- **Traces collected**: ~25,000 per byte (from 50,000 attempts)

## Files

```
aes_core.py           AES-128 + fault injection + temporal redundancy
generate_dataset.py   Collect ineffective-fault ciphertexts
traditional_sifa.py   Baseline: bit-bias key recovery
dl_sifa_model.py      Neural network key recovery (PyTorch MLP)
compare_results.py    Side-by-side comparison + plots
run_all.py            One command to run everything
report.pdf      report
```

## How to run

```bash
python run_all.py
```

## Results

Both methods successfully recover K10[0] = 0xD0.

| Method | Min traces | Correct score | Wrong score |
|--------|:----------:|:-------------:|:-----------:|
| Traditional SIFA | **20** | 1.0000 | 0.4980 |
| DL-SIFA (NN) | 50 | 1.0000 | 0.0001 |

- Traditional SIFA is faster (20 vs 50 traces)
- DL-SIFA has stronger separation (1.0 vs 0.0 instead of 1.0 vs 0.5)
- DL-SIFA advantage: doesn't need to know which bit was faulted

## Requirements

```bash
pip install numpy torch matplotlib
```

## References

1. Dobraunig et al., "SIFA: Exploiting ineffective fault inductions," TCHES 2018
2. Ramezanpour et al., "Fault intensity map analysis with neural network key distinguisher," ASHES 2019
