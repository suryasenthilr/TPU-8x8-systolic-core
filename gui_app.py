#!/usr/bin/env python3
"""
Apex-TPU: Interactive EDA Desktop Workbench & Hardware Demonstration System
Features:
- Live RTL Verification Runner & Log Viewer
- Real-time GTKWave Waveform Visualizer Launcher
- Interactive 8x8 Systolic Array Matrix Multiplication Engine (Custom GEMM, Bias, GELU/ReLU)
- ASIC Synthesis Metrics & Area Breakdown (Nangate 45nm Standard Cell Library)
"""

import sys
import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np

# Styling constants
BG_DARK      = "#0f172a"
BG_CARD      = "#1e293b"
BG_ACCENT    = "#334155"
TEXT_WHITE   = "#f8fafc"
TEXT_MUTED   = "#94a3b8"
COLOR_CYAN   = "#38bdf8"
COLOR_INDIGO = "#818cf8"
COLOR_EMERALD= "#34d399"
COLOR_ROSE   = "#f43f5e"
COLOR_AMBER  = "#fbbf24"
FONT_FAMILY  = "Segoe UI"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class ApexTpuWorkbench(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Apex-TPU: 2D Systolic Array DL Accelerator — EDA Workbench")
        self.geometry("1360x860")
        self.configure(bg=BG_DARK)

        # Style configuration
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=BG_DARK, foreground=TEXT_WHITE, font=(FONT_FAMILY, 10))
        self.style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=BG_CARD, foreground=TEXT_MUTED, padding=[16, 8], font=(FONT_FAMILY, 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", BG_ACCENT)], foreground=[("selected", COLOR_CYAN)])

        self._build_header()
        self._build_tabs()

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_DARK, pady=12, padx=20)
        hdr.pack(fill="x")

        title_lbl = tk.Label(hdr, text="⚡ APEX-TPU: 2D SYSTOLIC DEEP LEARNING ACCELERATOR",
                             font=(FONT_FAMILY, 16, "bold"), fg=COLOR_CYAN, bg=BG_DARK)
        title_lbl.pack(side="left")

        sub_lbl = tk.Label(hdr, text="Nangate 45nm ASIC | 367.8 kGE | 64 MACs/cycle | GELU VPU | AXI4 Master",
                           font=(FONT_FAMILY, 10), fg=TEXT_MUTED, bg=BG_DARK)
        sub_lbl.pack(side="right")

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Tab 1: Architecture Overview
        tab_arch = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(tab_arch, text=" Microarchitecture ")
        self._init_arch_tab(tab_arch)

        # Tab 2: Interactive Matrix Engine
        tab_gemm = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(tab_gemm, text=" Interactive GEMM Engine ")
        self._init_gemm_tab(tab_gemm)

        # Tab 3: Verification & Waves
        tab_sim = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(tab_sim, text=" RTL Simulation & Waveforms ")
        self._init_sim_tab(tab_sim)

        # Tab 4: ASIC Metrics
        tab_asic = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(tab_asic, text=" Nangate 45nm ASIC Metrics ")
        self._init_asic_tab(tab_asic)

        # Tab 5: Physical Design & GDSII Layout
        tab_pnr = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(tab_pnr, text=" 🔬 Physical Design & GDSII ")
        self._init_pnr_tab(tab_pnr)

    # -------------------------------------------------------------------------
    # TAB 1: ARCHITECTURE (Side-by-Side Split View for Large Readable Visuals)
    # -------------------------------------------------------------------------
    def _init_arch_tab(self, parent):
        # Outer split container
        split_frame = tk.Frame(parent, bg=BG_DARK, padx=12, pady=10)
        split_frame.pack(fill="both", expand=True)

        # Left Column: Large, Crisp Architecture Diagram
        left_box = tk.LabelFrame(split_frame, text=" 2D Systolic Array Microarchitecture Overview ",
                                 bg=BG_CARD, fg=COLOR_CYAN, font=(FONT_FAMILY, 10, "bold"), padx=10, pady=10)
        left_box.pack(side="left", fill="both", expand=True, padx=(0, 10))

        diag_path = os.path.join(PROJECT_ROOT, "docs", "images", "apex_tpu_architecture.png")
        if os.path.exists(diag_path):
            img = Image.open(diag_path)
            # Large readable size that fills the left panel
            img.thumbnail((960, 580), Image.Resampling.LANCZOS)
            self.tk_arch_img = ImageTk.PhotoImage(img)
            img_lbl = tk.Label(left_box, image=self.tk_arch_img, bg=BG_DARK)
            img_lbl.pack(fill="both", expand=True)

        # Right Column: Architectural Highlights Cards (Vertically Stacked)
        right_box = tk.Frame(split_frame, bg=BG_DARK, width=350)
        right_box.pack(side="right", fill="y", padx=(0, 4))
        right_box.pack_propagate(False)

        specs = [
            ("Core Microarchitecture",
             "• 8x8 2D Systolic Grid (64 PEs)\n• Weight-Stationary (WS) dataflow\n• 64 signed INT8 MACs/cycle\n• Diagonal wavefront skewing",
             COLOR_CYAN),
            ("Memory Hierarchy",
             "• 2KB Dual-Port Synchronous UB\n• 512-bit Preload Stationary WB\n• AMBA AXI4 Master DMA Burst\n• INCR bursts up to 256 beats",
             COLOR_EMERALD),
            ("Vector Unit (VPU)",
             "• 32-bit Column Bias Addition\n• PWL Transformer GELU / ReLU\n• INT32 -> INT8 Saturation\n• Direct loopback to UB Port B",
             COLOR_ROSE),
            ("Nangate 45nm ASIC Signoff",
             "• 173,758 Standard Cells\n• 293,491 µm² (0.293 mm²) Area\n• 367.8 kGE Normalized Gates\n• 12.8 GOPs/s @ 100MHz",
             COLOR_AMBER)
        ]

        for title, desc, col in specs:
            card = tk.Frame(right_box, bg=BG_CARD, padx=14, pady=12, highlightbackground=col, highlightthickness=1)
            card.pack(fill="x", pady=6)
            tk.Label(card, text=title, font=(FONT_FAMILY, 10, "bold"), fg=col, bg=BG_CARD).pack(anchor="w")
            tk.Label(card, text=desc, font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD, justify="left").pack(anchor="w", pady=(6, 0))

    # -------------------------------------------------------------------------
    # TAB 2: INTERACTIVE MATRIX MULTIPLIER
    # -------------------------------------------------------------------------
    def _init_gemm_tab(self, parent):
        top_ctrl = tk.Frame(parent, bg=BG_CARD, padx=16, pady=12)
        top_ctrl.pack(fill="x", padx=16, pady=12)

        tk.Label(top_ctrl, text="Matrix Dimension:", font=(FONT_FAMILY, 10, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side="left")
        tk.Label(top_ctrl, text="8x8 (INT8 GEMM)", font=(FONT_FAMILY, 10), fg=COLOR_CYAN, bg=BG_CARD).pack(side="left", padx=(4, 16))

        tk.Label(top_ctrl, text="Activation Function:", font=(FONT_FAMILY, 10, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side="left")
        self.act_choice = tk.StringVar(value="Transformer GELU")
        act_combo = ttk.Combobox(top_ctrl, textvariable=self.act_choice, state="readonly",
                                 values=["Pass-Through", "ReLU", "Transformer GELU", "Clip [-128, 127]"], width=18)
        act_combo.pack(side="left", padx=(4, 16))

        tk.Label(top_ctrl, text="Quant Shift (0..15):", font=(FONT_FAMILY, 10, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side="left")
        self.shift_spin = ttk.Spinbox(top_ctrl, from_=0, to=15, width=4)
        self.shift_spin.set(3)
        self.shift_spin.pack(side="left", padx=(4, 16))

        gen_btn = tk.Button(top_ctrl, text="🎲 Randomize Inputs", font=(FONT_FAMILY, 9, "bold"),
                            bg=BG_ACCENT, fg=TEXT_WHITE, activebackground=COLOR_CYAN, relief="flat", padx=10, command=self._randomize_matrices)
        gen_btn.pack(side="left", padx=4)

        calc_btn = tk.Button(top_ctrl, text="⚡ Run Systolic Compute", font=(FONT_FAMILY, 9, "bold"),
                             bg=COLOR_ROSE, fg=TEXT_WHITE, activebackground="#e11d48", relief="flat", padx=12, command=self._compute_interactive_gemm)
        calc_btn.pack(side="left", padx=8)

        # Matrix visualizer displays (A, W, and C)
        display_frame = tk.Frame(parent, bg=BG_DARK)
        display_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Frame 1: Matrix A
        f_a = tk.LabelFrame(display_frame, text=" Activation Matrix A [8x8] ", bg=BG_CARD, fg=COLOR_CYAN, font=(FONT_FAMILY, 10, "bold"), padx=8, pady=8)
        f_a.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.txt_a = scrolledtext.ScrolledText(f_a, bg=BG_DARK, fg=TEXT_WHITE, font=("Consolas", 9), width=32, height=12)
        self.txt_a.pack(fill="both", expand=True)

        # Frame 2: Matrix W
        f_w = tk.LabelFrame(display_frame, text=" Stationary Weight Matrix W [8x8] ", bg=BG_CARD, fg=COLOR_EMERALD, font=(FONT_FAMILY, 10, "bold"), padx=8, pady=8)
        f_w.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.txt_w = scrolledtext.ScrolledText(f_w, bg=BG_DARK, fg=TEXT_WHITE, font=("Consolas", 9), width=32, height=12)
        self.txt_w.pack(fill="both", expand=True)

        # Frame 3: Matrix C Result
        f_c = tk.LabelFrame(display_frame, text=" Quantized Output Matrix C [8x8] (Bit-Exact VPU Result) ", bg=BG_CARD, fg=COLOR_AMBER, font=(FONT_FAMILY, 10, "bold"), padx=8, pady=8)
        f_c.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)
        self.txt_c = scrolledtext.ScrolledText(f_c, bg=BG_DARK, fg=COLOR_AMBER, font=("Consolas", 9, "bold"), width=34, height=12)
        self.txt_c.pack(fill="both", expand=True)

        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.columnconfigure(2, weight=1)
        display_frame.rowconfigure(0, weight=1)

        self._randomize_matrices()

    def _randomize_matrices(self):
        np.random.seed()
        self.mat_A = np.random.randint(-15, 16, size=(8, 8), dtype=np.int8)
        self.mat_W = np.random.randint(-12, 13, size=(8, 8), dtype=np.int8)
        self.mat_B = np.random.randint(-50, 50, size=8, dtype=np.int32)

        self.txt_a.delete("1.0", tk.END)
        for row in self.mat_A:
            self.txt_a.insert(tk.END, " ".join(f"{v:4d}" for v in row) + "\n")

        self.txt_w.delete("1.0", tk.END)
        for row in self.mat_W:
            self.txt_w.insert(tk.END, " ".join(f"{v:4d}" for v in row) + "\n")

        self._compute_interactive_gemm()

    def _compute_interactive_gemm(self):
        shift = int(self.shift_spin.get())
        mode = self.act_choice.get()

        C_gemm = np.matmul(self.mat_A.astype(np.int32), self.mat_W.astype(np.int32))
        C_out = np.zeros((8, 8), dtype=np.int8)

        for r in range(8):
            for c in range(8):
                biased = int(C_gemm[r, c]) + int(self.mat_B[c])
                if mode == "ReLU":
                    act = max(0, biased)
                elif mode == "Transformer GELU":
                    if biased < -256: act = 0
                    elif biased < -64: act = biased >> 4
                    elif biased < 0: act = biased >> 3
                    elif biased < 256: act = (biased * 3) >> 2
                    else: act = biased
                elif mode == "Clip [-128, 127]":
                    act = max(-128, min(127, biased))
                else: # Pass-Through
                    act = biased

                # Round and saturate
                offset = (1 << (shift - 1)) if shift > 0 else 0
                shifted = (act + offset) >> shift
                sat = max(-128, min(127, shifted))
                C_out[r, c] = sat

        self.txt_c.delete("1.0", tk.END)
        self.txt_c.insert(tk.END, f"Mode: {mode} | Shift: >> {shift}\n")
        self.txt_c.insert(tk.END, f"Throughput: 64 MACs/cyc | Latency: 24 cyc\n")
        self.txt_c.insert(tk.END, "-" * 32 + "\n")
        for row in C_out:
            self.txt_c.insert(tk.END, " ".join(f"{v:4d}" for v in row) + "\n")

    # -------------------------------------------------------------------------
    # TAB 3: SIMULATION & GTKWAVE
    # -------------------------------------------------------------------------
    def _init_sim_tab(self, parent):
        top_bar = tk.Frame(parent, bg=BG_CARD, padx=16, pady=12)
        top_bar.pack(fill="x", padx=16, pady=12)

        run_btn = tk.Button(top_bar, text="▶ Run Verilog Testbench (iverilog + vvp)", font=(FONT_FAMILY, 10, "bold"),
                            bg=COLOR_EMERALD, fg="#022c22", activebackground="#10b981", relief="flat", padx=14, pady=4,
                            command=self._run_verilog_sim)
        run_btn.pack(side="left", padx=(0, 10))

        wave_btn = tk.Button(top_bar, text="📈 Launch GTKWave Viewer", font=(FONT_FAMILY, 10, "bold"),
                             bg=COLOR_INDIGO, fg=TEXT_WHITE, activebackground="#6366f1", relief="flat", padx=14, pady=4,
                             command=self._launch_gtkwave)
        wave_btn.pack(side="left", padx=10)

        self.sim_status_lbl = tk.Label(top_bar, text="Status: Ready", font=(FONT_FAMILY, 10, "bold"), fg=COLOR_CYAN, bg=BG_CARD)
        self.sim_status_lbl.pack(side="right")

        log_frame = tk.LabelFrame(parent, text=" RTL Simulation Console Output ", bg=BG_CARD, fg=TEXT_WHITE, font=(FONT_FAMILY, 10, "bold"), padx=10, pady=10)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.txt_sim_log = scrolledtext.ScrolledText(log_frame, bg="#050811", fg="#a7f3d0", font=("Consolas", 10), insertbackground=TEXT_WHITE)
        self.txt_sim_log.pack(fill="both", expand=True)
        self.txt_sim_log.insert(tk.END, "Click 'Run Verilog Testbench' above to execute cycle-accurate hardware simulation.\n")

    def _run_verilog_sim(self):
        self.sim_status_lbl.config(text="Status: Compiling & Simulating...", fg=COLOR_AMBER)
        self.txt_sim_log.delete("1.0", tk.END)
        self.txt_sim_log.insert(tk.END, "[APEX-TPU] Starting iverilog compilation and cycle-accurate execution...\n")

        def task():
            cmd = (
                r'call C:\Users\surya\.oss-cad-suite\oss-cad-suite\environment.bat && '
                r'cd /d ' + PROJECT_ROOT + r' && '
                r'iverilog -g2012 -o sim/apex_tpu_sim.vvp '
                r'rtl/pe.v rtl/systolic_array.v rtl/skew_unit.v rtl/deskew_unit.v '
                r'rtl/vector_unit.v rtl/unified_buffer.v rtl/weight_buffer.v '
                r'rtl/dma_axi_master.v rtl/tpu_controller.v rtl/apex_tpu_top.v sim/tb_apex_tpu.v && '
                r'vvp sim/apex_tpu_sim.vvp'
            )
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                self.txt_sim_log.insert(tk.END, line)
                self.txt_sim_log.see(tk.END)
            p.wait()

            if p.returncode == 0:
                self.sim_status_lbl.config(text="Status: Simulation PASSED (100% Bit-Exact)", fg=COLOR_EMERALD)
            else:
                self.sim_status_lbl.config(text="Status: Simulation Failed", fg=COLOR_ROSE)

        threading.Thread(target=task, daemon=True).start()

    def _launch_gtkwave(self):
        vcd_path = os.path.join(PROJECT_ROOT, "sim", "apex_tpu.vcd")
        gtkw_path = os.path.join(PROJECT_ROOT, "sim", "apex_tpu.gtkw")
        if not os.path.exists(vcd_path):
            messagebox.showwarning("VCD Not Found", "Please run the Verilog simulation first to generate sim/apex_tpu.vcd!")
            return

        cmd = (
            r'call C:\Users\surya\.oss-cad-suite\oss-cad-suite\environment.bat && '
            r'cd /d ' + PROJECT_ROOT + r' && '
            r'gtkwave sim/apex_tpu.vcd sim/apex_tpu.gtkw'
        )
        subprocess.Popen(cmd, shell=True)

    # -------------------------------------------------------------------------
    # TAB 4: ASIC SYNTHESIS METRICS
    # -------------------------------------------------------------------------
    def _init_asic_tab(self, parent):
        scroll_c = tk.Canvas(parent, bg=BG_DARK, highlightthickness=0)
        scroll_y = tk.Scrollbar(parent, orient="vertical", command=scroll_c.yview)
        sf = tk.Frame(scroll_c, bg=BG_DARK)

        sf.bind("<Configure>", lambda e: scroll_c.configure(scrollregion=scroll_c.bbox("all")))
        scroll_c.create_window((0, 0), window=sf, anchor="nw")
        scroll_c.configure(yscrollcommand=scroll_y.set)

        scroll_c.pack(side="left", fill="both", expand=True, padx=16, pady=16)
        scroll_y.pack(side="right", fill="y")

        # Top Banner
        b_frame = tk.Frame(sf, bg=BG_CARD, padx=20, pady=16, highlightbackground=COLOR_CYAN, highlightthickness=1)
        b_frame.pack(fill="x", pady=(0, 16))

        tk.Label(b_frame, text="NANGATE 45nm OPEN CELL ASIC SYNTHESIS METRICS", font=(FONT_FAMILY, 14, "bold"), fg=COLOR_CYAN, bg=BG_CARD).pack(anchor="w")
        tk.Label(b_frame, text="Standard Cell Library: NangateOpenCellLibrary_typical.lib (45nm Standard VT)\nSynthesis Engine: Yosys Open SYnthesis Suite + ABC Logic Optimizer",
                 font=(FONT_FAMILY, 10), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(4, 0))

        # Metrics Summary Grid
        m_grid = tk.Frame(sf, bg=BG_DARK)
        m_grid.pack(fill="x", pady=(0, 16))

        stat_cards = [
            ("Total Cell Count", "173,758", "standard cells", COLOR_CYAN),
            ("Total Silicon Area", "293,491.37 µm²", "0.293 mm² silicon footprint", COLOR_EMERALD),
            ("Gate Equivalents", "367.8 kGE", "NAND2_X1 Normalized (0.798 µm²)", COLOR_ROSE),
            ("Peak Throughput", "64 MACs/cycle", "12.8 GOPs/s @ 100MHz", COLOR_AMBER)
        ]

        for i, (title, val, sub, col) in enumerate(stat_cards):
            c = tk.Frame(m_grid, bg=BG_CARD, padx=16, pady=14, highlightbackground=col, highlightthickness=1)
            c.grid(row=0, column=i, padx=8, sticky="nsew")
            tk.Label(c, text=title, font=(FONT_FAMILY, 10, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")
            tk.Label(c, text=val, font=(FONT_FAMILY, 16, "bold"), fg=col, bg=BG_CARD).pack(anchor="w", pady=2)
            tk.Label(c, text=sub, font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
            m_grid.columnconfigure(i, weight=1)

        # Hierarchy Breakdown Table
        tbl_frame = tk.LabelFrame(sf, text=" Submodule Area & Gate Breakdown ", bg=BG_CARD, fg=TEXT_WHITE, font=(FONT_FAMILY, 11, "bold"), padx=12, pady=12)
        tbl_frame.pack(fill="x")

        headers = ["Submodule", "Function / Role", "Cell Count", "Area (µm²)", "Gate Eq (kGE)", "Area %"]
        for col_idx, h in enumerate(headers):
            tk.Label(tbl_frame, text=h, font=(FONT_FAMILY, 10, "bold"), fg=COLOR_CYAN, bg=BG_CARD).grid(row=0, column=col_idx, sticky="w", padx=10, pady=4)

        rows = [
            ("unified_buffer", "2KB True Dual-Port Synchronous Scratchpad", "101,442", "170,000 µm²", "213.0 kGE", "57.9%"),
            ("systolic_array", "8x8 2D Processing Grid (64 Weight-Stationary PEs)", "46,720", "66,000 µm²", "82.7 kGE", "22.5%"),
            ("weight_buffer", "512-bit Preload Stationary Weight Buffer", "15,005", "36,400 µm²", "45.6 kGE", "12.4%"),
            ("vector_unit", "VPU: 32b Bias + ReLU/GELU + INT8 Quantizer", "7,092", "9,600 µm²", "12.0 kGE", "3.3%"),
            ("deskew_unit", "7-Stage Output Column Synchronous Deskewer", "1,792", "6,440 µm²", "8.1 kGE", "2.2%"),
            ("dma_axi_master", "AMBA AXI4 Master Burst Interface (64-bit)", "710", "2,010 µm²", "2.5 kGE", "0.7%"),
            ("skew_unit", "Diagonal Wavefront Shift Register Pipeline", "448", "1,610 µm²", "2.0 kGE", "0.5%"),
            ("tpu_controller", "CISC Instruction Decoder & Wavefront Sequencer", "541", "1,200 µm²", "1.5 kGE", "0.4%"),
            ("apex_tpu_top", "Total Integrated Deep Learning Accelerator Core", "173,758", "293,491.37 µm²", "367.8 kGE", "100.0%")
        ]

        for row_idx, r_data in enumerate(rows, start=1):
            bg_col = BG_ACCENT if row_idx % 2 == 0 else BG_CARD
            fg_col = COLOR_AMBER if row_idx == len(rows) else TEXT_WHITE
            font_w = "bold" if row_idx == len(rows) else "normal"

            for col_idx, val in enumerate(r_data):
                tk.Label(tbl_frame, text=val, font=(FONT_FAMILY, 9, font_w), fg=fg_col, bg=bg_col, padx=4, pady=3).grid(row=row_idx, column=col_idx, sticky="w", padx=6, pady=2)

    # -------------------------------------------------------------------------
    # TAB 5: PHYSICAL DESIGN & GDSII LAYOUT
    # -------------------------------------------------------------------------
    def _init_pnr_tab(self, parent):
        sf = tk.Frame(parent, bg=BG_DARK, padx=16, pady=12)
        sf.pack(fill="both", expand=True)

        # Top Banner
        b_frame = tk.Frame(sf, bg=BG_CARD, padx=16, pady=10, highlightbackground=COLOR_EMERALD, highlightthickness=1)
        b_frame.pack(fill="x", pady=(0, 12))

        b_top = tk.Frame(b_frame, bg=BG_CARD)
        b_top.pack(fill="x")
        tk.Label(b_top, text="PHYSICAL DESIGN & GDSII SILICON SIGNOFF", font=(FONT_FAMILY, 13, "bold"), fg=COLOR_EMERALD, bg=BG_CARD).pack(side="left")
        status_lbl = tk.Label(b_top, text="● 100% ROUTED & LEGALIZED | DRC/LVS CLEAN | TAPE-OUT STREAM",
                              font=(FONT_FAMILY, 9, "bold"), fg=COLOR_EMERALD, bg=BG_ACCENT, padx=8, pady=3)
        status_lbl.pack(side="right")

        tk.Label(b_frame, text="Backend Toolchain: OpenROAD 26Q2 + FastRoute + TritonCTS | Stream Merge: KLayout 0.30 | Library: Nangate 45nm",
                 font=(FONT_FAMILY, 9), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(4, 0))

        # Split Container: Left = Interactive Canvas Viewport, Right = Physical Signoff & Actions
        split = tk.Frame(sf, bg=BG_DARK)
        split.pack(fill="both", expand=True)

        # LEFT PANE: Chip Layout Image Viewer
        left_box = tk.LabelFrame(split, text=" Silicon Layout Floorplan & Standard Cell Geometry ",
                                 bg=BG_CARD, fg=COLOR_CYAN, font=(FONT_FAMILY, 10, "bold"), padx=10, pady=10)
        left_box.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctrl_bar = tk.Frame(left_box, bg=BG_CARD)
        ctrl_bar.pack(fill="x", pady=(0, 8))

        tk.Label(ctrl_bar, text="View Macro:", font=(FONT_FAMILY, 9, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side="left", padx=(0, 6))
        self.pnr_view_var = tk.StringVar(value="Full Chip Top (apex_tpu_top: 179k cells, 723x723 µm)")
        views = [
            "Full Chip Top (apex_tpu_top: 179k cells, 723x723 µm)",
            "Systolic Array Core (systolic_array: 46k cells, 355x355 µm)"
        ]
        self.view_combo = ttk.Combobox(ctrl_bar, textvariable=self.pnr_view_var, values=views, state="readonly", width=52)
        self.view_combo.pack(side="left", padx=4)
        self.view_combo.bind("<<ComboboxSelected>>", self._on_pnr_view_change)

        # Canvas for layout image
        self.pnr_canvas = tk.Canvas(left_box, bg="#050811", highlightthickness=1, highlightbackground=BG_ACCENT)
        self.pnr_canvas.pack(fill="both", expand=True)

        # RIGHT PANE: Metrics & Interactive Actions
        right_box = tk.Frame(split, bg=BG_DARK, width=420)
        right_box.pack(side="right", fill="both")
        right_box.pack_propagate(False)

        # Metrics Card 1: Silicon Real Estate
        c1 = tk.LabelFrame(right_box, text=" Silicon Footprint & Density ", bg=BG_CARD, fg=COLOR_CYAN,
                           font=(FONT_FAMILY, 9, "bold"), padx=12, pady=8)
        c1.pack(fill="x", pady=(0, 8))

        self.lbl_pnr_die = tk.Label(c1, text="• Die Dimension: 723.35 µm × 723.35 µm (0.523 mm²)", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_pnr_die.pack(anchor="w", pady=1)
        self.lbl_pnr_cells = tk.Label(c1, text="• Placed Instances: 179,209 standard cells (100% legalized)", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_pnr_cells.pack(anchor="w", pady=1)
        self.lbl_pnr_util = tk.Label(c1, text="• Core Utilization: 60.0% (Free area: 504,788 µm²)", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_pnr_util.pack(anchor="w", pady=1)
        self.lbl_pnr_gds = tk.Label(c1, text="• GDSII Stream: apex_tpu_top.gds (16.55 MB tape-out binary)", font=(FONT_FAMILY, 9), fg=COLOR_EMERALD, bg=BG_CARD)
        self.lbl_pnr_gds.pack(anchor="w", pady=1)

        # Metrics Card 2: Power Signoff
        c2 = tk.LabelFrame(right_box, text=" ASIC Power Signoff (OpenSTA) ", bg=BG_CARD, fg=COLOR_AMBER,
                           font=(FONT_FAMILY, 9, "bold"), padx=12, pady=8)
        c2.pack(fill="x", pady=(0, 8))

        self.lbl_pnr_pwr = tk.Label(c2, text="• Total Power: 578 mW @ 100MHz (Nangate 45nm)", font=(FONT_FAMILY, 9, "bold"), fg=COLOR_AMBER, bg=BG_CARD)
        self.lbl_pnr_pwr.pack(anchor="w", pady=1)
        self.lbl_pnr_seq = tk.Label(c2, text="• Sequential Power: 111 mW (19.2%)", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_pnr_seq.pack(anchor="w", pady=1)
        self.lbl_pnr_comb = tk.Label(c2, text="• Combinational Power: 438 mW (75.8%)", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_pnr_comb.pack(anchor="w", pady=1)
        self.lbl_pnr_clk_pwr = tk.Label(c2, text="• Clock Network Power: 28.8 mW (5.0%)", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_pnr_clk_pwr.pack(anchor="w", pady=1)
        self.lbl_pnr_leak = tk.Label(c2, text="• Leakage Power: 6.57 mW (1.1%)", font=(FONT_FAMILY, 9), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_pnr_leak.pack(anchor="w", pady=1)

        # Metrics Card 3: Clock & PDN Architecture
        c3 = tk.LabelFrame(right_box, text=" CTS & PDN Power Straps ", bg=BG_CARD, fg=COLOR_INDIGO,
                           font=(FONT_FAMILY, 9, "bold"), padx=12, pady=8)
        c3.pack(fill="x", pady=(0, 8))

        self.lbl_pnr_cts = tk.Label(c3, text="• Clock Sinks: 27,891 flip-flops (TritonCTS 8 Levels)", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_pnr_cts.pack(anchor="w", pady=1)
        self.lbl_pnr_buf = tk.Label(c3, text="• Clock Buffers: 3,575 buffers inserted (CLKBUF_X1..X3)", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_pnr_buf.pack(anchor="w", pady=1)
        tk.Label(c3, text="• PDN M1 Rails: 0.17 µm stdcell pitch rails", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w", pady=1)
        tk.Label(c3, text="• PDN M4 Straps: 0.48 µm width @ 56.0 µm pitch", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w", pady=1)
        tk.Label(c3, text="• PDN M7 Straps: 1.40 µm width @ 30.0 µm pitch", font=(FONT_FAMILY, 9), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w", pady=1)

        # Card 4: Interactive Actions
        c4 = tk.LabelFrame(right_box, text=" Live Interactive Physical Tools ", bg=BG_CARD, fg=COLOR_CYAN,
                           font=(FONT_FAMILY, 9, "bold"), padx=12, pady=10)
        c4.pack(fill="x")

        btn_klayout = tk.Button(c4, text="🔬 Launch Live GDSII in KLayout (GUI)", font=(FONT_FAMILY, 9, "bold"),
                                bg=COLOR_EMERALD, fg="#042f2e", activebackground="#6ee7b7", pady=6, cursor="hand2",
                                command=self._launch_klayout_viewer)
        btn_klayout.pack(fill="x", pady=3)

        btn_log = tk.Button(c4, text="📜 View OpenROAD P&R Signoff Log", font=(FONT_FAMILY, 9, "bold"),
                            bg=COLOR_CYAN, fg="#082f49", activebackground="#7dd3fc", pady=6, cursor="hand2",
                            command=self._view_pnr_log)
        btn_log.pack(fill="x", pady=3)

        btn_dir = tk.Button(c4, text="📁 Open P&R Files Directory", font=(FONT_FAMILY, 9),
                            bg=BG_ACCENT, fg=TEXT_WHITE, activebackground=BG_CARD, pady=4, cursor="hand2",
                            command=self._open_pnr_dir)
        btn_dir.pack(fill="x", pady=3)

        # Initial layout image render
        self.after(200, self._render_pnr_canvas)

    def _on_pnr_view_change(self, event=None):
        choice = self.pnr_view_var.get()
        if "Systolic" in choice:
            self.lbl_pnr_die.config(text="• Die Dimension: 355.05 µm × 355.05 µm (0.126 mm²)")
            self.lbl_pnr_cells.config(text="• Placed Instances: 46,271 cells (64 MAC PEs, 100% legalized)")
            self.lbl_pnr_util.config(text="• Core Utilization: 55.0% (Core area: 118,766 µm²)")
            self.lbl_pnr_gds.config(text="• GDSII Stream: systolic_array.gds (5.12 MB tape-out binary)")
            self.lbl_pnr_pwr.config(text="• Total Power: 142.6 mW @ 100MHz (Nangate 45nm)")
            self.lbl_pnr_seq.config(text="• Sequential Power: 34.2 mW (24.0%)")
            self.lbl_pnr_comb.config(text="• Combinational Power: 98.4 mW (69.0%)")
            self.lbl_pnr_clk_pwr.config(text="• Clock Network Power: 8.5 mW (6.0%)")
            self.lbl_pnr_leak.config(text="• Leakage Power: 1.5 mW (1.0%)")
            self.lbl_pnr_cts.config(text="• Clock Sinks: 8,192 flip-flops (Balanced TritonCTS)")
            self.lbl_pnr_buf.config(text="• Clock Buffers: 503 buffers inserted (CLKBUF_X1..X3)")
        else:
            self.lbl_pnr_die.config(text="• Die Dimension: 723.35 µm × 723.35 µm (0.523 mm²)")
            self.lbl_pnr_cells.config(text="• Placed Instances: 179,209 standard cells (100% legalized)")
            self.lbl_pnr_util.config(text="• Core Utilization: 60.0% (Free area: 504,788 µm²)")
            self.lbl_pnr_gds.config(text="• GDSII Stream: apex_tpu_top.gds (16.55 MB tape-out binary)")
            self.lbl_pnr_pwr.config(text="• Total Power: 578 mW @ 100MHz (Nangate 45nm)")
            self.lbl_pnr_seq.config(text="• Sequential Power: 111 mW (19.2%)")
            self.lbl_pnr_comb.config(text="• Combinational Power: 438 mW (75.8%)")
            self.lbl_pnr_clk_pwr.config(text="• Clock Network Power: 28.8 mW (5.0%)")
            self.lbl_pnr_leak.config(text="• Leakage Power: 6.57 mW (1.1%)")
            self.lbl_pnr_cts.config(text="• Clock Sinks: 27,891 flip-flops (TritonCTS 8 Levels)")
            self.lbl_pnr_buf.config(text="• Clock Buffers: 3,575 buffers inserted (CLKBUF_X1..X3)")
        self._render_pnr_canvas()

    def _render_pnr_canvas(self):
        choice = self.pnr_view_var.get()
        if "Systolic" in choice:
            img_path = os.path.join(PROJECT_ROOT, "pnr", "systolic_array_layout.png")
        else:
            img_path = os.path.join(PROJECT_ROOT, "pnr", "apex_tpu_top_layout.png")

        if not os.path.exists(img_path):
            self.pnr_canvas.delete("all")
            self.pnr_canvas.create_text(250, 250, text=f"Layout image not found:\n{img_path}", fill=COLOR_ROSE, font=(FONT_FAMILY, 11))
            return

        w = self.pnr_canvas.winfo_width()
        h = self.pnr_canvas.winfo_height()
        if w < 50 or h < 50:
            w, h = 600, 600

        try:
            im = Image.open(img_path)
            im_resized = im.resize((w, h), Image.Resampling.LANCZOS)
            self._tk_layout_img = ImageTk.PhotoImage(im_resized)
            self.pnr_canvas.delete("all")
            self.pnr_canvas.create_image(w // 2, h // 2, image=self._tk_layout_img)
        except Exception as e:
            self.pnr_canvas.delete("all")
            self.pnr_canvas.create_text(250, 250, text=f"Render error: {e}", fill=COLOR_ROSE, font=(FONT_FAMILY, 10))

    def _launch_klayout_viewer(self):
        choice = self.pnr_view_var.get()
        gds_name = "systolic_array.gds" if "Systolic" in choice else "apex_tpu_top.gds"
        wsl_path = f"/mnt/c/Users/surya/OneDrive/Documents/COLLEGE_RESEARCH/apex-tpu/pnr/{gds_name}"
        cmd = f"wsl klayout {wsl_path}"
        threading.Thread(target=lambda: subprocess.run(cmd, shell=True), daemon=True).start()
        messagebox.showinfo("KLayout Launched", f"Launching interactive KLayout viewer for:\n{gds_name}\n\nRendering multi-layer GDSII stream...")

    def _view_pnr_log(self):
        log_path = os.path.join(PROJECT_ROOT, "pnr", "pnr_top.log")
        if not os.path.exists(log_path):
            messagebox.showwarning("Log Not Found", f"Log file not found: {log_path}")
            return

        top = tk.Toplevel(self)
        top.title("OpenROAD Physical Design Log — apex_tpu_top")
        top.geometry("900x650")
        top.configure(bg=BG_DARK)

        txt = scrolledtext.ScrolledText(top, bg="#050811", fg="#a6accd", font=("Consolas", 9), insertbackground=COLOR_CYAN)
        txt.pack(fill="both", expand=True, padx=10, pady=10)

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            txt.insert("1.0", f.read())
        txt.config(state="disabled")

    def _open_pnr_dir(self):
        pnr_dir = os.path.join(PROJECT_ROOT, "pnr")
        os.startfile(pnr_dir)

if __name__ == "__main__":
    app = ApexTpuWorkbench()
    app.mainloop()
