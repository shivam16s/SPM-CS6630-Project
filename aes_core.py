
"""
AES-128 with Temporal Redundancy + Round 8 Fault Injection
SPM CS6630 - Final Project
Roll Number : CS25M048
"""

import os
import numpy as np

# ---- AES S-BOX (FIPS-197) ----

SBOX = [
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16,
]

INV_SBOX = [0] * 256
for i, v in enumerate(SBOX):
    INV_SBOX[v] = i

RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]

# ---- GF(2^8) multiply ----

def gf_mult(a, b):
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        carry = a & 0x80
        a = (a << 1) & 0xFF
        if carry:
            a ^= 0x1B
        b >>= 1
    return result

# precompute tables
GF = {}
for c in [2, 3, 9, 11, 13, 14]:
    GF[c] = [gf_mult(c, x) for x in range(256)]


# ---- AES round ops ----

def sub_bytes(s):    return [SBOX[b] for b in s]
def inv_sub_bytes(s): return [INV_SBOX[b] for b in s]

def shift_rows(s):
    r = list(s)
    r[1], r[5], r[9],  r[13] = s[5],  s[9],  s[13], s[1]
    r[2], r[6], r[10], r[14] = s[10], s[14], s[2],  s[6]
    r[3], r[7], r[11], r[15] = s[15], s[3],  s[7],  s[11]
    return r

def inv_shift_rows(s):
    r = list(s)
    r[1], r[5], r[9],  r[13] = s[13], s[1],  s[5],  s[9]
    r[2], r[6], r[10], r[14] = s[10], s[14], s[2],  s[6]
    r[3], r[7], r[11], r[15] = s[7],  s[11], s[15], s[3]
    return r

def mix_columns(s):
    out = [0]*16
    for c in range(4):
        i = c*4
        out[i]   = GF[2][s[i]] ^ GF[3][s[i+1]] ^ s[i+2]       ^ s[i+3]
        out[i+1] = s[i]        ^ GF[2][s[i+1]] ^ GF[3][s[i+2]] ^ s[i+3]
        out[i+2] = s[i]        ^ s[i+1]        ^ GF[2][s[i+2]] ^ GF[3][s[i+3]]
        out[i+3] = GF[3][s[i]] ^ s[i+1]        ^ s[i+2]        ^ GF[2][s[i+3]]
    return out

def inv_mix_columns(s):
    out = [0]*16
    for c in range(4):
        i = c*4
        out[i]   = GF[14][s[i]] ^ GF[11][s[i+1]] ^ GF[13][s[i+2]] ^ GF[9][s[i+3]]
        out[i+1] = GF[9][s[i]]  ^ GF[14][s[i+1]] ^ GF[11][s[i+2]] ^ GF[13][s[i+3]]
        out[i+2] = GF[13][s[i]] ^ GF[9][s[i+1]]  ^ GF[14][s[i+2]] ^ GF[11][s[i+3]]
        out[i+3] = GF[11][s[i]] ^ GF[13][s[i+1]] ^ GF[9][s[i+2]]  ^ GF[14][s[i+3]]
    return out

def add_round_key(s, rk):
    return [a ^ b for a, b in zip(s, rk)]


# ---- Key schedule ----

def key_expansion(key):
    w = []
    for i in range(4):
        w.append(list(key[i*4:(i+1)*4]))
    for i in range(4, 44):
        temp = list(w[i-1])
        if i % 4 == 0:
            temp = temp[1:] + temp[:1]
            temp = [SBOX[b] for b in temp]
            temp[0] ^= RCON[i//4 - 1]
        w.append([w[i-4][j] ^ temp[j] for j in range(4)])
    rks = []
    for r in range(11):
        rk = []
        for j in range(4):
            rk.extend(w[r*4 + j])
        rks.append(rk)
    return rks


# ---- Encryption ----

def aes_encrypt(pt, rks):
    s = add_round_key(list(pt), rks[0])
    for r in range(1, 10):
        s = sub_bytes(s)
        s = shift_rows(s)
        s = mix_columns(s)
        s = add_round_key(s, rks[r])
    # round 10 (no mix columns)
    s = sub_bytes(s)
    s = shift_rows(s)
    s = add_round_key(s, rks[10])
    return s


# ---- fault injection at round 8 input ----

def aes_with_fault(pt, rks, fbyte, fbit):
    # rounds 1-7
    s = add_round_key(list(pt), rks[0])
    for r in range(1, 8):
        s = sub_bytes(s)
        s = shift_rows(s)
        s = mix_columns(s)
        s = add_round_key(s, rks[r])

    # inject stuck-at-0 fault before round 8 SubBytes
    clean = list(s)
    faulted = list(s)
    faulted[fbyte] = faulted[fbyte] & (~(1 << fbit) & 0xFF)
    ineffective = (clean[fbyte] == faulted[fbyte])

    # rounds 8-10
    def do_remaining(st):
        for r in range(8, 10):
            st = sub_bytes(st)
            st = shift_rows(st)
            st = mix_columns(st)
            st = add_round_key(st, rks[r])
        st = sub_bytes(st)
        st = shift_rows(st)
        st = add_round_key(st, rks[10])
        return st

    return do_remaining(faulted), do_remaining(clean), ineffective


# ---- Temporal redundancy check ----
# encrypt twice, if outputs differ then fault was detected -> suppress

def protected_encrypt(pt, rks, fbyte=None, fbit=None):
    if fbyte is None:
        return aes_encrypt(pt, rks), False
    fct, cct, ineff = aes_with_fault(pt, rks, fbyte, fbit)
    if fct == cct:
        return cct, False   # fault not detected -> passes
    else:
        return None, True   # fault detected -> suppressed


# ---- NIST test vector ----

if __name__ == "__main__":
    key = [0x2B,0x7E,0x15,0x16,0x28,0xAE,0xD2,0xA6,
           0xAB,0xF7,0x15,0x88,0x09,0xCF,0x4F,0x3C]
    pt  = [0x32,0x43,0xF6,0xA8,0x88,0x5A,0x30,0x8D,
           0x31,0x31,0x98,0xA2,0xE0,0x37,0x07,0x34]
    expected = [0x39,0x25,0x84,0x1D,0x02,0xDC,0x09,0xFB,
                0xDC,0x11,0x85,0x97,0x19,0x6A,0x0B,0x32]
    rks = key_expansion(key)
    ct = aes_encrypt(pt, rks)
    if ct == expected:
        print("AES-128 test: PASS")
    else:
        print("AES-128 test: FAIL")
        print(f"  Expected: {[hex(x) for x in expected]}")
        print(f"  Got:      {[hex(x) for x in ct]}")
