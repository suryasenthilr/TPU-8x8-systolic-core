#!/usr/bin/env python3
"""
Apex-TPU Microarchitecture Diagram Generator
Creates high-resolution publication-quality diagram of the 2D Systolic Tensor Processing Unit.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_diagram():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('#0f172a') # Slate-900 dark theme
    ax.set_facecolor('#0f172a')

    # Color palette
    c_ctrl   = '#38bdf8' # Light Sky Blue
    c_dma    = '#818cf8' # Indigo
    c_sram   = '#34d399' # Emerald Green
    c_skew   = '#fbbf24' # Amber
    c_array  = '#f43f5e' # Rose / Red
    c_pe     = '#fb7185' # Lighter Rose
    c_deskew = '#f59e0b' # Amber/Orange
    c_vpu    = '#a855f7' # Purple
    c_bus    = '#94a3b8' # Slate
    c_text   = '#f8fafc' # White

    # Title Banner
    ax.text(8.0, 9.5, "Apex-TPU: 2D Systolic Tensor Processing Unit Architecture",
            fontsize=20, fontweight='bold', color=c_text, ha='center', va='center')
    ax.text(8.0, 9.15, "Weight-Stationary Dataflow | AMBA AXI4 DMA | 64-MAC Systolic Grid | Transformer GELU VPU | Nangate 45nm ASIC",
            fontsize=11, color='#94a3b8', ha='center', va='center')

    # 1. Host Interface & CISC Controller
    p_ctrl = patches.FancyBboxPatch((0.5, 6.6), 3.2, 2.0, boxstyle="round,pad=0.15", ec=c_ctrl, fc='#1e293b', lw=2)
    ax.add_patch(p_ctrl)
    ax.text(2.1, 8.35, "CISC TPU CONTROLLER", fontsize=12, fontweight='bold', color=c_ctrl, ha='center')
    ax.text(2.1, 7.9, "• Instruction Decoder (32-bit)\n• OP_LOAD_W / IN / MATMUL\n• OP_CONFIG_VPU / STORE\n• Wavefront Cycle Sequencer",
            fontsize=9.5, color=c_text, ha='center', va='center')

    # 2. AXI4 Master DMA Controller
    p_dma = patches.FancyBboxPatch((0.5, 3.8), 3.2, 2.3, boxstyle="round,pad=0.15", ec=c_dma, fc='#1e293b', lw=2)
    ax.add_patch(p_dma)
    ax.text(2.1, 5.85, "AMBA AXI4 MASTER DMA", fontsize=12, fontweight='bold', color=c_dma, ha='center')
    ax.text(2.1, 4.9, "• 64-bit Read/Write Channels\n• Autonomous Burst Engine\n• INCR Burst (1-256 beats)\n• System DRAM Interface",
            fontsize=9.5, color=c_text, ha='center', va='center')

    # External DRAM box
    p_dram = patches.FancyBboxPatch((0.5, 1.2), 3.2, 1.8, boxstyle="round,pad=0.15", ec='#64748b', fc='#1e293b', lw=1.5, ls='--')
    ax.add_patch(p_dram)
    ax.text(2.1, 2.3, "EXTERNAL SYSTEM MEMORY", fontsize=11, fontweight='bold', color='#cbd5e1', ha='center')
    ax.text(2.1, 1.7, "DDR4 / HBM DRAM\n(Shared System Physical Space)", fontsize=9, color='#94a3b8', ha='center')

    # 3. On-Chip Buffers
    # Unified Buffer (UB)
    p_ub = patches.FancyBboxPatch((4.3, 5.2), 2.8, 3.4, boxstyle="round,pad=0.15", ec=c_sram, fc='#1e293b', lw=2)
    ax.add_patch(p_ub)
    ax.text(5.7, 8.35, "UNIFIED BUFFER (UB)", fontsize=12, fontweight='bold', color=c_sram, ha='center')
    ax.text(5.7, 6.8, "2KB True Dual-Port SRAM\n(256 rows x 64 bits)\n\n• Port A: DMA R/W Stream\n• Port B: Array Feed & VPU Write",
            fontsize=9.5, color=c_text, ha='center', va='center')

    # Weight Buffer (WB)
    p_wb = patches.FancyBboxPatch((4.3, 1.2), 2.8, 3.4, boxstyle="round,pad=0.15", ec=c_sram, fc='#1e293b', lw=2)
    ax.add_patch(p_wb)
    ax.text(5.7, 4.35, "WEIGHT BUFFER (WB)", fontsize=12, fontweight='bold', color=c_sram, ha='center')
    ax.text(5.7, 2.8, "Tiled Weight Storage\n(64 rows x 64 bits)\n\n• 8 Weight Tiles Cached\n• 512-bit Parallel Latch\n• 1-Cycle Stationary Reload",
            fontsize=9.5, color=c_text, ha='center', va='center')

    # 4. Skew Unit
    p_skew = patches.FancyBboxPatch((7.7, 5.2), 1.6, 3.4, boxstyle="round,pad=0.1", ec=c_skew, fc='#1e293b', lw=2)
    ax.add_patch(p_skew)
    ax.text(8.5, 8.35, "SKEW UNIT", fontsize=11, fontweight='bold', color=c_skew, ha='center')
    ax.text(8.5, 6.8, "Wavefront\nSkewing\n\nRow 0: 0 cyc\nRow 1: 1 cyc\nRow 2: 2 cyc\n...\nRow 7: 7 cyc",
            fontsize=9, color=c_text, ha='center', va='center')

    # 5. 2D Systolic Array (8x8 Grid)
    p_sa = patches.FancyBboxPatch((10.0, 3.6), 5.5, 5.0, boxstyle="round,pad=0.2", ec=c_array, fc='#1e293b', lw=2.5)
    ax.add_patch(p_sa)
    ax.text(12.75, 8.35, "8x8 2D SYSTOLIC PROCESSING GRID (64 PEs)", fontsize=13, fontweight='bold', color=c_array, ha='center')

    # Mini PE grid visualization (4x4 representative inside 8x8)
    for r in range(4):
        for c in range(4):
            x = 10.6 + c * 1.15
            y = 7.1 - r * 1.05
            p_pe = patches.Rectangle((x, y), 0.95, 0.75, ec=c_pe, fc='#334155', lw=1.2)
            ax.add_patch(p_pe)
            ax.text(x + 0.475, y + 0.375, f"PE\n{r},{c}", fontsize=7.5, color='#ffffff', ha='center', va='center', fontweight='bold')

    ax.text(12.75, 4.0, "[Weight-Stationary Dataflow: 64 MACs/Cycle | 8-bit Signed Multipliers | 32-bit Accumulators]",
            fontsize=9, color='#fca5a5', ha='center', va='center', fontstyle='italic')

    # 6. Deskew Unit
    p_deskew = patches.FancyBboxPatch((10.0, 1.8), 2.5, 1.4, boxstyle="round,pad=0.1", ec=c_deskew, fc='#1e293b', lw=2)
    ax.add_patch(p_deskew)
    ax.text(11.25, 2.9, "DESKEW UNIT", fontsize=11, fontweight='bold', color=c_deskew, ha='center')
    ax.text(11.25, 2.3, "Col j delayed by (7-j) cyc\nParallel Row Realignment",
            fontsize=8.5, color=c_text, ha='center', va='center')

    # 7. Vector Processing Unit (VPU)
    p_vpu = patches.FancyBboxPatch((13.0, 1.2), 2.5, 2.0, boxstyle="round,pad=0.15", ec=c_vpu, fc='#1e293b', lw=2)
    ax.add_patch(p_vpu)
    ax.text(14.25, 2.95, "VECTOR UNIT (VPU)", fontsize=11, fontweight='bold', color=c_vpu, ha='center')
    ax.text(14.25, 2.05, "• 32-bit Bias Addition\n• ReLU / Transformer GELU\n• INT32 -> INT8 Quantizer\n• Clamping [-128, +127]",
            fontsize=8.5, color=c_text, ha='center', va='center')

    # Interconnect Arrows
    def draw_arrow(x1, y1, x2, y2, color, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=7, headlength=7))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.15, label, fontsize=8, color=color, ha='center', fontweight='bold')

    # DRAM <-> DMA
    draw_arrow(2.1, 3.0, 2.1, 3.8, c_dma, "AXI4 Burst")
    draw_arrow(2.1, 3.8, 2.1, 3.0, c_dma)

    # Controller -> Modules
    draw_arrow(2.1, 6.6, 2.1, 6.1, c_ctrl, "DMA Trigger")
    draw_arrow(3.7, 7.6, 4.3, 7.6, c_ctrl, "Read/Write")
    draw_arrow(3.7, 7.1, 7.7, 7.1, c_ctrl, "En")
    draw_arrow(3.7, 6.8, 10.0, 6.8, c_ctrl, "Compute En")

    # DMA -> Buffers
    draw_arrow(3.7, 5.5, 4.3, 5.5, c_dma, "Activations")
    draw_arrow(3.7, 4.2, 4.3, 2.9, c_dma, "Weights")

    # UB -> Skew -> Systolic Array
    draw_arrow(7.1, 6.9, 7.7, 6.9, c_sram, "64b Act")
    draw_arrow(9.3, 6.9, 10.0, 6.9, c_skew, "Wavefront")

    # WB -> Systolic Array (Stationary Weights)
    draw_arrow(7.1, 2.9, 10.0, 4.5, c_sram, "512b Stationary Weights")

    # Systolic Array -> Deskew -> VPU
    draw_arrow(11.25, 3.6, 11.25, 3.2, c_array, "256b Partial Sums")
    draw_arrow(12.5, 2.5, 13.0, 2.5, c_deskew, "Aligned")

    # VPU Feedback -> UB Port B (Loopback for deep networks)
    ax.annotate('', xy=(5.7, 5.2), xytext=(14.25, 1.2),
                arrowprops=dict(facecolor=c_vpu, edgecolor=c_vpu, width=2, headwidth=7, headlength=7,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(10.0, 0.6, "Direct Layer-to-Layer Activation Writeback (INT8 Quantized Vector)",
            fontsize=9.5, color=c_vpu, ha='center', fontweight='bold')

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "apex_tpu_architecture.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Diagram saved successfully to {out_path}")

if __name__ == "__main__":
    create_diagram()
