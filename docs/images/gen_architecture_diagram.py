#!/usr/bin/env python3
"""
Apex-TPU High-Legibility Microarchitecture Diagram Generator
Uses balanced dimensions and high-contrast, bold typography so all text
is razor-sharp and clearly legible inside desktop GUI viewers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_diagram():
    # 15x9 inches at 150 DPI produces a clean, high-density 2250x1350 image
    fig, ax = plt.subplots(figsize=(15, 9.2), dpi=150)
    fig.patch.set_facecolor('#0f172a') # Slate-900 dark theme
    ax.set_facecolor('#0f172a')

    # Color palette
    c_ctrl   = '#38bdf8' # Light Sky Blue
    c_dma    = '#818cf8' # Indigo
    c_sram   = '#34d399' # Emerald Green
    c_skew   = '#fbbf24' # Amber
    c_array  = '#f43f5e' # Rose
    c_pe     = '#fb7185' # Light Rose
    c_deskew = '#f59e0b' # Amber/Orange
    c_vpu    = '#c084fc' # Purple / Violet
    c_text   = '#ffffff' # Pure White

    # Title Banner
    ax.text(7.5, 8.8, "Apex-TPU: 2D Systolic Tensor Processing Unit Architecture",
            fontsize=17, fontweight='bold', color=c_text, ha='center', va='center')
    ax.text(7.5, 8.45, "Weight-Stationary Dataflow | AMBA AXI4 DMA | 64-MAC Systolic Grid | Transformer GELU VPU | Nangate 45nm ASIC",
            fontsize=10.5, color='#cbd5e1', ha='center', va='center', fontweight='bold')

    # 1. CISC Controller
    p_ctrl = patches.FancyBboxPatch((0.4, 6.0), 3.0, 2.1, boxstyle="round,pad=0.12", ec=c_ctrl, fc='#1e293b', lw=2.2)
    ax.add_patch(p_ctrl)
    ax.text(1.9, 7.75, "CISC TPU CONTROLLER", fontsize=11, fontweight='bold', color=c_ctrl, ha='center')
    ax.text(1.9, 6.9, "• Instruction Decoder (32-bit)\n• OP_LOAD_W / IN / MATMUL\n• OP_CONFIG_VPU / STORE\n• Wavefront Cycle Sequencer",
            fontsize=9.5, color=c_text, ha='center', va='center', fontweight='bold')

    # 2. AXI4 Master DMA
    p_dma = patches.FancyBboxPatch((0.4, 3.4), 3.0, 2.3, boxstyle="round,pad=0.12", ec=c_dma, fc='#1e293b', lw=2.2)
    ax.add_patch(p_dma)
    ax.text(1.9, 5.35, "AMBA AXI4 MASTER DMA", fontsize=11, fontweight='bold', color=c_dma, ha='center')
    ax.text(1.9, 4.4, "• 64-bit Read/Write Channels\n• Autonomous Burst Engine\n• INCR Burst (1-256 beats)\n• System DRAM Interface",
            fontsize=9.5, color=c_text, ha='center', va='center', fontweight='bold')

    # External DRAM
    p_dram = patches.FancyBboxPatch((0.4, 1.0), 3.0, 1.9, boxstyle="round,pad=0.12", ec='#64748b', fc='#1e293b', lw=1.8, ls='--')
    ax.add_patch(p_dram)
    ax.text(1.9, 2.3, "EXTERNAL SYSTEM MEMORY", fontsize=10.5, fontweight='bold', color='#cbd5e1', ha='center')
    ax.text(1.9, 1.7, "DDR4 / HBM DRAM\n(Shared System Physical Space)", fontsize=9, color='#94a3b8', ha='center', fontweight='bold')

    # 3. Unified Buffer
    p_ub = patches.FancyBboxPatch((3.9, 4.8), 2.7, 3.3, boxstyle="round,pad=0.12", ec=c_sram, fc='#1e293b', lw=2.2)
    ax.add_patch(p_ub)
    ax.text(5.25, 7.75, "UNIFIED BUFFER (UB)", fontsize=11, fontweight='bold', color=c_sram, ha='center')
    ax.text(5.25, 6.3, "2KB True Dual-Port SRAM\n(256 rows x 64 bits)\n\n• Port A: DMA R/W Stream\n• Port B: Array Feed / VPU In",
            fontsize=9.5, color=c_text, ha='center', va='center', fontweight='bold')

    # Weight Buffer
    p_wb = patches.FancyBboxPatch((3.9, 1.0), 2.7, 3.3, boxstyle="round,pad=0.12", ec=c_sram, fc='#1e293b', lw=2.2)
    ax.add_patch(p_wb)
    ax.text(5.25, 3.95, "WEIGHT BUFFER (WB)", fontsize=11, fontweight='bold', color=c_sram, ha='center')
    ax.text(5.25, 2.5, "Tiled Weight Storage\n(64 rows x 64 bits)\n\n• 8 Weight Tiles Cached\n• 512-bit Parallel Latch\n• 1-Cycle Stationary Reload",
            fontsize=9.5, color=c_text, ha='center', va='center', fontweight='bold')

    # 4. Skew Unit
    p_skew = patches.FancyBboxPatch((7.1, 4.8), 1.6, 3.3, boxstyle="round,pad=0.1", ec=c_skew, fc='#1e293b', lw=2.2)
    ax.add_patch(p_skew)
    ax.text(7.9, 7.75, "SKEW UNIT", fontsize=11, fontweight='bold', color=c_skew, ha='center')
    ax.text(7.9, 6.3, "Wavefront\nDelay Lines\n\nRow 0: 0 cyc\nRow 1: 1 cyc\nRow 2: 2 cyc\n...\nRow 7: 7 cyc",
            fontsize=9, color=c_text, ha='center', va='center', fontweight='bold')

    # 5. 2D Systolic Processing Grid (8x8)
    p_sa = patches.FancyBboxPatch((9.2, 3.3), 5.4, 4.8, boxstyle="round,pad=0.15", ec=c_array, fc='#1e293b', lw=2.5)
    ax.add_patch(p_sa)
    ax.text(11.9, 7.75, "8x8 2D SYSTOLIC PROCESSING GRID (64 PEs)", fontsize=12, fontweight='bold', color=c_array, ha='center')

    # PE Grid (4x4 representative sample)
    for r in range(4):
        for c in range(4):
            x = 9.8 + c * 1.12
            y = 6.6 - r * 0.98
            p_pe = patches.Rectangle((x, y), 0.94, 0.72, ec=c_pe, fc='#334155', lw=1.3)
            ax.add_patch(p_pe)
            ax.text(x + 0.47, y + 0.36, f"PE\n{r},{c}", fontsize=8.5, color='#ffffff', ha='center', va='center', fontweight='bold')

    ax.text(11.9, 3.65, "[Weight-Stationary: 64 MACs/Cycle | 8-bit Signed Multipliers | 32-bit Accumulators]",
            fontsize=8.5, color='#fca5a5', ha='center', va='center', fontweight='bold')

    # 6. Deskew Unit
    p_deskew = patches.FancyBboxPatch((9.2, 1.4), 2.5, 1.5, boxstyle="round,pad=0.1", ec=c_deskew, fc='#1e293b', lw=2.2)
    ax.add_patch(p_deskew)
    ax.text(10.45, 2.5, "DESKEW UNIT", fontsize=10.5, fontweight='bold', color=c_deskew, ha='center')
    ax.text(10.45, 1.9, "Col j delayed by (7-j) cyc\nParallel Row Realignment",
            fontsize=8.5, color=c_text, ha='center', va='center', fontweight='bold')

    # 7. Vector Unit (VPU)
    p_vpu = patches.FancyBboxPatch((12.1, 1.0), 2.5, 1.9, boxstyle="round,pad=0.12", ec=c_vpu, fc='#1e293b', lw=2.2)
    ax.add_patch(p_vpu)
    ax.text(13.35, 2.55, "VECTOR UNIT (VPU)", fontsize=10.5, fontweight='bold', color=c_vpu, ha='center')
    ax.text(13.35, 1.75, "• 32-bit Bias Addition\n• ReLU / Transformer GELU\n• INT32 -> INT8 Quantizer\n• Clamping [-128, +127]",
            fontsize=8, color=c_text, ha='center', va='center', fontweight='bold')

    # Interconnect Arrows with bold labels
    def draw_arrow(x1, y1, x2, y2, color, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=2.2, headwidth=7.5, headlength=7.5))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.14, label, fontsize=8.5, color=color, ha='center', fontweight='bold')

    draw_arrow(1.9, 2.9, 1.9, 3.4, c_dma, "AXI4")
    draw_arrow(1.9, 3.4, 1.9, 2.9, c_dma)

    draw_arrow(1.9, 6.0, 1.9, 5.7, c_ctrl, "Trigger")
    draw_arrow(3.4, 7.1, 3.9, 7.1, c_ctrl, "R/W")
    draw_arrow(3.4, 6.6, 7.1, 6.6, c_ctrl, "En")
    draw_arrow(3.4, 6.3, 9.2, 6.3, c_ctrl, "Compute En")

    draw_arrow(3.4, 5.0, 3.9, 5.0, c_dma, "Act In")
    draw_arrow(3.4, 3.8, 3.9, 2.6, c_dma, "Weights")

    draw_arrow(6.6, 6.4, 7.1, 6.4, c_sram, "64b Act")
    draw_arrow(8.7, 6.4, 9.2, 6.4, c_skew, "Wavefront")

    draw_arrow(6.6, 2.6, 9.2, 4.3, c_sram, "512b Stationary Weights")

    draw_arrow(10.45, 3.3, 10.45, 2.9, c_array, "256b Partial Sums")
    draw_arrow(11.7, 2.15, 12.1, 2.15, c_deskew, "Aligned")

    ax.annotate('', xy=(5.25, 4.8), xytext=(13.35, 1.0),
                arrowprops=dict(facecolor=c_vpu, edgecolor=c_vpu, width=2.2, headwidth=7.5, headlength=7.5,
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(9.3, 0.45, "Direct Layer-to-Layer Activation Writeback (INT8 Quantized Vector)",
            fontsize=9.5, color=c_vpu, ha='center', fontweight='bold')

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "apex_tpu_architecture.png")
    plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"High-legibility diagram saved to {out_path}")

if __name__ == "__main__":
    create_diagram()
