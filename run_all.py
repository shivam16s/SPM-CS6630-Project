
"""
Run the full DL-SIFA pipeline
SPM CS6630 - Final Project
Roll Number : CS25M048
"""

import os
import time

def main():
    start = time.time()
    print("=" * 60)
    print("DL-SIFA Project - Full Pipeline")
    print("=" * 60)

    # step 1: verify AES
    print("\n[1] AES verification")
    from aes_core import key_expansion, aes_encrypt
    key = [0x2B,0x7E,0x15,0x16,0x28,0xAE,0xD2,0xA6,
           0xAB,0xF7,0x15,0x88,0x09,0xCF,0x4F,0x3C]
    pt  = [0x32,0x43,0xF6,0xA8,0x88,0x5A,0x30,0x8D,
           0x31,0x31,0x98,0xA2,0xE0,0x37,0x07,0x34]
    expected = [0x39,0x25,0x84,0x1D,0x02,0xDC,0x09,0xFB,
                0xDC,0x11,0x85,0x97,0x19,0x6A,0x0B,0x32]
    rks = key_expansion(key)
    ct = aes_encrypt(pt, rks)
    assert ct == expected, "AES test failed!"
    print("  PASS")

    # step 2: generate dataset
    print("\n[2] Dataset generation")
    if not os.path.exists(os.path.join("dataset", "sifa_dataset.npz")):
        from generate_dataset import main as gen_main
        gen_main()
    else:
        print("  Dataset exists, skipping.")

    # step 3: traditional SIFA
    print("\n[3] Traditional SIFA")
    from traditional_sifa import main as trad_main
    trad_main()

    # step 4: DL-SIFA
    print("\n[4] DL-SIFA")
    from dl_sifa_model import main as dl_main
    dl_main()

    # step 5: comparison plots
    print("\n[5] Comparison plots")
    from compare_results import main as comp_main
    comp_main()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
