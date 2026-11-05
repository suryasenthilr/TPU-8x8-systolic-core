#!/usr/bin/env python3
"""
Apex-TPU Golden Model & Verification Vector Generator
Implements bit-exact mathematical model for:
- INT8 Matrix Multiplication (8x8 GEMM)
- Bias Addition
- Piecewise Linear Transformer GELU / ReLU
- Fixed-Point Quantization with Rounding and Saturation
Generates memory images and test vectors for tb_apex_tpu.v.
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
    """Bit-exact match to vector_unit.v rounding & saturation"""
    offset = (1 << (shift - 1)) if shift > 0 else 0
    shifted = (val + offset) >> shift
    if shifted > 127:
        return 127
    elif shifted < -128:
        return -128
    else:
        return shifted

def generate_test_case():
    np.random.seed(42)
    ARRAY_SIZE = 8

    # 1. Generate Input Activations (8x8, signed INT8)
    # Using representative dynamic range for deep learning activations
    A = np.random.randint(-15, 16, size=(ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)

    # 2. Generate Weight Matrix (8x8, signed INT8)
    W = np.random.randint(-12, 13, size=(ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)

    # 3. Generate Bias Vector (8 elements, signed INT32)
    bias = np.random.randint(-50, 50, size=ARRAY_SIZE, dtype=np.int32)

    # 4. Standard Matrix Multiplication: C = A @ W
    # Note: A is rows x K, W is K x cols. Here 8x8 @ 8x8
    C_gemm = np.matmul(A.astype(np.int32), W.astype(np.int32))

    # 5. Golden VPU Execution for test cases:
    # Mode 1: ReLU with shift=2
    shift_relu = 2
    C_relu_out = np.zeros((ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)
    for r in range(ARRAY_SIZE):
        for c in range(ARRAY_SIZE):
            biased = int(C_gemm[r, c]) + int(bias[c])
            act = relu(biased)
            C_relu_out[r, c] = quantize_int8(act, shift_relu)

    # Mode 2: Transformer GELU with shift=3
    shift_gelu = 3
    C_gelu_out = np.zeros((ARRAY_SIZE, ARRAY_SIZE), dtype=np.int8)
    for r in range(ARRAY_SIZE):
        for c in range(ARRAY_SIZE):
            biased = int(C_gemm[r, c]) + int(bias[c])
            act = gelu_pwl(biased)
            C_gelu_out[r, c] = quantize_int8(act, shift_gelu)

    print("=" * 60)
    print("Apex-TPU Golden Model Generated Successfully!")
    print(f"Matrix Size: {ARRAY_SIZE}x{ARRAY_SIZE} (INT8)")
    print(f"Activation Matrix A (Row 0): {A[0, :]}")
    print(f"Weight Matrix W (Row 0):     {W[0, :]}")
    print(f"Bias Vector:                 {bias}")
    print(f"GEMM Unbiased C[0, 0..3]:    {C_gemm[0, :4]}")
    print(f"GELU Quantized Output (Row 0): {C_gelu_out[0, :]}")
    print("=" * 60)

    # Write out hex data for Verilog readmemh
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # DRAM memory file:
    # Offset 0x00..0x07 (8 rows): Weights (row 0 to 7, 8 bytes per row = 64-bit hex)
    # Offset 0x10..0x17 (8 rows): Inputs  (row 0 to 7, 8 bytes per row = 64-bit hex)
    # Offset 0x20..0x27 (8 rows): Output results buffer in DRAM
    with open(os.path.join(out_dir, "dram_init.hex"), "w") as f:
        # Write 8 rows of weights at address 0..7
        for r in range(ARRAY_SIZE):
            row_bytes = [int(w) & 0xFF for w in W[r, :]]
            val64 = 0
            for i, b in enumerate(row_bytes):
                val64 |= (b << (i * 8))
            f.write(f"// Weight Row {r}\n{val64:016x}\n")

        # Fill up to row 16
        for _ in range(8):
            f.write("0000000000000000\n")

        # Write 8 rows of activations at address 16..23
        for r in range(ARRAY_SIZE):
            row_bytes = [int(a) & 0xFF for a in A[r, :]]
            val64 = 0
            for i, b in enumerate(row_bytes):
                val64 |= (b << (i * 8))
            f.write(f"// Input Row {r}\n{val64:016x}\n")

    # Expected outputs hex files
    with open(os.path.join(out_dir, "expected_gelu.hex"), "w") as f:
        for r in range(ARRAY_SIZE):
            row_bytes = [int(c) & 0xFF for c in C_gelu_out[r, :]]
            val64 = 0
            for i, b in enumerate(row_bytes):
                val64 |= (b << (i * 8))
            f.write(f"{val64:016x}\n")

    # Bias hex file
    with open(os.path.join(out_dir, "bias.hex"), "w") as f:
        for b in bias:
            f.write(f"{int(b) & 0xFFFFFFFF:08x}\n")

    print(f"Test vectors written to {out_dir}")

if __name__ == "__main__":
    generate_test_case()
