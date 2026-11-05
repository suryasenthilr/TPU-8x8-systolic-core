# Apex-TPU Microarchitecture Specification

Apex-TPU is a cycle-accurate, synthesizable 2D systolic Deep Learning Accelerator (NPU/TPU) designed for high-efficiency integer matrix multiplication (GEMM) and Transformer inference. The microarchitecture implements a Weight-Stationary (WS) dataflow inspired by the Google TPU family (TPUv1–v4), coupled with an on-chip Unified Activation Buffer (UB), an autonomous AMBA AXI4 Master DMA burst engine, a CISC matrix instruction sequencer, and a Vector Processing Unit (VPU) supporting piece-wise linear Transformer GELU and symmetric INT8 quantization.

---

## 1. Systolic Array Execution & Wavefront Timing

The compute core contains an $8 \times 8$ grid of 64 Processing Elements (PEs). Each PE integrates an 8-bit signed multiplier, a 32-bit partial sum accumulator register, and stationary weight storage.

```
                    Activations (A) -> [Skew Unit]
                                         |
                                         v  Row 0 (t=0)
       +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0  | PE 0,0 | PE 0,1 | PE 0,2 | PE 0,3 | PE 0,4 | PE 0,5 | PE 0,6 | PE 0,7 |
       +--------+--------+--------+--------+--------+--------+--------+--------+
                                         |
                                         v  Row 1 (t=1)
       +--------+--------+--------+--------+--------+--------+--------+--------+
Row 1  | PE 1,0 | PE 1,1 | PE 1,2 | PE 1,3 | PE 1,4 | PE 1,5 | PE 1,6 | PE 1,7 |
       +--------+--------+--------+--------+--------+--------+--------+--------+
                                         ...
       +--------+--------+--------+--------+--------+--------+--------+--------+
Row 7  | PE 7,0 | PE 7,1 | PE 7,2 | PE 7,3 | PE 7,4 | PE 7,5 | PE 7,6 | PE 7,7 |
       +--------+--------+--------+--------+--------+--------+--------+--------+
                                         |
                                [Deskew Unit]
                                         |
                                [Vector Unit (VPU)]
                                         |
                             Quantized INT8 Results -> [Unified Buffer]
```

### 1.1 Weight-Stationary Spatial Dataflow

In Weight-Stationary execution, weights remain pinned in PE registers during matrix computation:

$$\mathbf{C} = \mathbf{A} \times \mathbf{W}$$

- **Weight Preload**: Prior to matrix execution, weights $\mathbf{W} \in \mathbb{Z}^{8 \times 8}$ are loaded from system memory via DMA into the Weight Buffer, then latched simultaneously into all 64 PEs in 1 clock cycle via `OP_SET_WEIGHT`.
- **Horizontal Activation Forwarding**: Activations $\mathbf{A}$ stream from the left, propagating horizontally from $\text{PE}_{r, c}$ to $\text{PE}_{r, c+1}$ through registered registers.
- **Vertical Partial Sum Accumulation**: Column partial sums accumulate downwards from $\text{PE}_{r, c}$ to $\text{PE}_{r+1, c}$. The top inputs ($\text{PE}_{0, c}$) receive 0, and bottom outputs ($\text{PE}_{7, c}$) emerge as fully accumulated 32-bit sums.

### 1.2 Skew and Deskew Timing Synchronization

Because activations propagate horizontally by 1 PE per cycle and partial sums propagate vertically by 1 PE per cycle, inputs must enter along a diagonal wavefront:

1. **Input Skewing**:
   Row $r$ activations are delayed by $r$ clock cycles using shift registers:
   $$t_{\text{in}}(r) = t_{\text{base}} + r \quad (r = 0, \dots, 7)$$

2. **PE Computation Time**:
   The computation at row $r$, column $c$ occurs at cycle:
   $$t_{\text{comp}}(r, c) = t_{\text{base}} + 1 + r + c$$

3. **Bottom Array Emergence**:
   Column $c$ partial sums emerge from the bottom PE ($r=7$) at cycle:
   $$t_{\text{emerge}}(c) = t_{\text{base}} + 1 + 7 + c + 1 = t_{\text{base}} + 9 + c$$

4. **Output Deskewing**:
   Column $c$ is delayed by $(ARRAY\_SIZE - 1 - c) = (7 - c)$ clock cycles:
   $$t_{\text{deskew}}(c) = (t_{\text{base}} + 9 + c) + (7 - c) = t_{\text{base}} + 16$$
   All 8 columns align synchronously into a single parallel row vector at cycle $t_{\text{base}} + 16$.

5. **VPU Pipeline Latency**:
   The Vector Unit registers the aligned sum, applies bias addition, activation, and quantization, completing at cycle $t_{\text{base}} + 17$. The total latency for an $8 \times 8 \times 8$ GEMM tile is exactly 24 cycles.

---

## 2. Vector Processing Unit (VPU)

The Vector Unit operates on 8 parallel 32-bit channels emerging from the deskew unit:

### 2.1 Bias Vector Addition
Each column sum adds an element from the 32-bit bias vector:
$$y_i = s_i + b_i$$

### 2.2 Activation Function Pipeline

Apex-TPU supports four programmable activation modes:

1. **Bypass (`2'b00`)**: Linear output with no non-linear activation.
2. **ReLU (`2'b01`)**: Rectified Linear Unit:
   $$\text{ReLU}(y) = \max(0, y)$$
3. **Transformer GELU (`2'b10`)**: Gaussian Error Linear Unit implemented via a 5-region piecewise linear (PWL) approximation:
   $$\text{GELU}_{\text{PWL}}(y) = \begin{cases} 0, & y < -256 \\ y \gg 4, & -256 \le y < -64 \\ y \gg 3, & -64 \le y < 0 \\ (3y) \gg 2, & 0 \le y < 256 \\ y, & y \ge 256 \end{cases}$$
4. **Clip (`2'b11`)**: Bounds clamping to $[-128, 127]$.

### 2.3 Dynamic INT8 Quantization & Saturation
To allow direct feedback into subsequent Transformer layers without floating-point conversion, the 32-bit values are scaled using programmable right-shift scaling with symmetric rounding and saturation:

$$\text{offset} = \begin{cases} 1 \ll (\text{shift} - 1), & \text{shift} > 0 \\ 0, & \text{shift} = 0 \end{cases}$$

$$\hat{y} = (y + \text{offset}) \gg \text{shift}$$

$$y_{\text{INT8}} = \min(127, \max(-128, \hat{y}))$$

---

## 3. CISC Instruction Set Architecture (ISA)

Apex-TPU utilizes 32-bit CISC instructions dispatched from a host CPU:

| Opcode | Mnemonic | Description | Operand Fields |
| :--- | :--- | :--- | :--- |
| `0x0` | `OP_NOP` | No operation | None |
| `0x1` | `OP_LOAD_WEIGHT` | DRAM $\to$ Weight Buffer via DMA | `[23:16]` DRAM row, `[15:8]` WB row, `[7:0]` length |
| `0x2` | `OP_LOAD_INPUT` | DRAM $\to$ Unified Buffer via DMA | `[23:16]` DRAM row, `[15:8]` UB row, `[7:0]` length |
| `0x3` | `OP_SET_WEIGHT` | Latch stationary weights into PEs | `[26:24]` Tile Index (0..7) |
| `0x4` | `OP_MATMUL` | Stream UB $\to$ Array $\to$ VPU $\to$ UB | `[23:16]` UB read row, `[15:8]` UB write row, `[7:0]` K |
| `0x5` | `OP_CONFIG_VPU` | Configure VPU parameters | `[25:24]` Act Func, `[20]` Bias En, `[19:16]` Quant Shift |
| `0x6` | `OP_STORE_OUT` | Unified Buffer $\to$ DRAM via DMA | `[23:16]` DRAM row, `[15:8]` UB row, `[7:0]` length |
| `0xF` | `OP_HALT` | Stop execution and assert interrupt | None |

---

## 4. Nangate 45nm ASIC Synthesis Metrics

The design was synthesized using Yosys and ABC against the standard Open Cell Nangate 45nm library (`NangateOpenCellLibrary_typical.lib`).

```
=============================================================================
Apex-TPU Nangate 45nm Standard Cell Synthesis Summary
=============================================================================
Top-Level Module:            apex_tpu_top
Standard Cell Count:         173,758 cells
Silicon Core Area:           293,491.37 µm² (0.293 mm²)
Sequential Cell Area:        121,954.62 µm² (41.55%)
Flip-Flop Count:             26,015 FFs (20,608 DFF_X1 + 5,406 DFFR_X1 + 1 DFFS_X1)
NAND2_X1 Equivalent Area:    0.798 µm²
Gate Equivalents (GE):       367,783.7 GE (~367.8 kGE)
Clock Period Target:         10.0 ns (100 MHz)
Peak Compute Throughput:     64 MACs/cycle (12.8 GOPs/s @ 100 MHz)
=============================================================================
```

### Submodule Area Breakdown

| Module | Purpose | Cells | Area ($\mu m^2$) | Gate Equivalents (kGE) | % of Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `unified_buffer` | 2KB Dual-Port Synchronous SRAM | 101,442 | 170,000.0 | 213.0 kGE | 57.9% |
| `systolic_array` | 64 Weight-Stationary PEs | 46,720 | 66,000.0 | 82.7 kGE | 22.5% |
| `weight_buffer` | 512-bit Tile Shadow Registers + RAM | 15,005 | 36,400.0 | 45.6 kGE | 12.4% |
| `vector_unit` | VPU: Bias, GELU/ReLU, INT8 Quant | 7,092 | 9,600.0 | 12.0 kGE | 3.3% |
| `deskew_unit` | 7-Stage Parallel Column Deskewer | 1,792 | 6,440.0 | 8.1 kGE | 2.2% |
| `dma_axi_master` | 64-bit AMBA AXI4 Master Controller | 710 | 2,010.0 | 2.5 kGE | 0.7% |
| `skew_unit` | Diagonal Wavefront Skew Pipeline | 448 | 1,610.0 | 2.0 kGE | 0.5% |
| `tpu_controller` | CISC Instruction Decoder & FSM | 541 | 1,200.0 | 1.5 kGE | 0.4% |
| **Total** | **Integrated Apex-TPU Core** | **173,758** | **293,491.37** | **367.8 kGE** | **100.0%** |
