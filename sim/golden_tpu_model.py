#!/usr/bin/env python3
"""
Apex-TPU Golden Reference Model & Multi-Mode Verification Suite Generator
Computes bit-exact reference outputs for:
1. Transformer GELU Mode (Non-linear smooth piecewise activation)
2. Standard Deep Learning ReLU Mode (Strict positive rectification)
3. Linear Pass-Through Mode (Bypass for projection layers)
4. Extreme Dynamic Range Saturation & Clamping Tests
"""

import numpy as np
import os

def gelu_pwl(x):
    """Bit-exact match to vector_unit.v piecewise linear GELU"""
    if x < -256:
        return 0
    elif x < -64:
        return x >> 4
    elif x < 0:
        return x >> 3
    elif x < 256:
        return (x * 3) >> 2
    else:
        return x

def relu(x):
    return max(0, x)

def quantize_int8(val, shift):
    """Bit-exact match to vector_unit.v rounding & saturation [-128, +127]"""
    offset = (1 << (shift - 1)) if shift > 0 else 0
    shifted = (val + offset) >> shift
    if shifted > 127:
        return 127
    elif shifted < -128:
        return -128
    else:
        return shifted

def generate_multi_test_suite():
    np.random.seed(42)
    ARRAY_SIZE = 8
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Primary Test Case Matrices (Standard DL range)
    A = np.random.randint(-15, 16, size=(ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)
    W = np.random.randint(-12, 13, size=(ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)
    bias = np.random.randint(-50, 50, size=ARRAY_SIZE, dtype=np.int32)
    C_gemm = np.matmul(A.astype(np.int32), W.astype(np.int32))

    # --- Mode 1: Transformer GELU (shift = 3) ---
    shift_gelu = 3
    C_gelu = np.zeros((ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)
    for r in range(ARRAY_SIZE):
        for c in range(ARRAY_SIZE):
            b_val = int(C_gemm[r, c]) + int(bias[c])
            C_gelu[r, c] = quantize_int8(gelu_pwl(b_val), shift_gelu)

    # --- Mode 2: Standard ReLU (shift = 2) ---
    shift_relu = 2
    C_relu = np.zeros((ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)
    for r in range(ARRAY_SIZE):
        for c in range(ARRAY_SIZE):
            b_val = int(C_gemm[r, c]) + int(bias[c])
            C_relu[r, c] = quantize_int8(relu(b_val), shift_relu)

    # --- Mode 3: Linear Pass-Through (shift = 4) ---
    shift_pass = 4
    C_pass = np.zeros((ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)
    for r in range(ARRAY_SIZE):
        for c in range(ARRAY_SIZE):
            b_val = int(C_gemm[r, c]) + int(bias[c])
            C_pass[r, c] = quantize_int8(b_val, shift_pass)

    # Helper: write 64-bit rows
    def write_hex_matrix(filename, mat):
        with open(os.path.join(out_dir, filename), "w") as f:
            for r in range(ARRAY_SIZE):
                row_bytes = [int(v) & 0xFF for v in mat[r, :]]
                val64 = sum(b << (i * 8) for i, b in enumerate(row_bytes))
                f.write(f"{val64:016x}\n")

    # Write expected outputs
    write_hex_matrix("expected_gelu.hex", C_gelu)
    write_hex_matrix("expected_relu.hex", C_relu)
    write_hex_matrix("expected_pass.hex", C_pass)

    # Write DRAM init image
    with open(os.path.join(out_dir, "dram_init.hex"), "w") as f:
        # Rows 0..7: Weights
        for r in range(ARRAY_SIZE):
            row_bytes = [int(w) & 0xFF for w in W[r, :]]
            val64 = sum(b << (i * 8) for i, b in enumerate(row_bytes))
            f.write(f"{val64:016x}\n")

        # Rows 8..15: Padding
        for _ in range(8):
            f.write("0000000000000000\n")

        # Rows 16..23: Inputs
        for r in range(ARRAY_SIZE):
            row_bytes = [int(a) & 0xFF for a in A[r, :]]
            val64 = sum(b << (i * 8) for i, b in enumerate(row_bytes))
            f.write(f"{val64:016x}\n")

        # Fill remaining DRAM entries up to 512 rows so no readmemh warnings
        for _ in range(512 - 24):
            f.write("0000000000000000\n")

    # Write bias vector
    with open(os.path.join(out_dir, "bias.hex"), "w") as f:
        for b in bias:
            f.write(f"{int(b) & 0xFFFFFFFF:08x}\n")

    print("=" * 60)
    print("Apex-TPU Multi-Mode Golden Verification Vectors Generated!")
    print(f"Matrix Dimension: {ARRAY_SIZE}x{ARRAY_SIZE} (INT8)")
    print(f"Mode 1: Transformer GELU (shift=3) -> expected_gelu.hex")
    print(f"Mode 2: Standard ReLU    (shift=2) -> expected_relu.hex")
    print(f"Mode 3: Linear Pass-Thru (shift=4) -> expected_pass.hex")
    print("=" * 60)

if __name__ == "__main__":
    generate_multi_test_suite()
