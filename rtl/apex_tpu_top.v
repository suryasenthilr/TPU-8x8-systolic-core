// =============================================================================
// File: apex_tpu_top.v
// Module: apex_tpu_top
// Description: Apex-TPU Top-Level Systolic Tensor Processing Unit Core.
//              Integrates:
//              - CISC TPU Instruction Controller & Sequencer
//              - AMBA AXI4 Master DMA Burst Interface (64-bit bus)
//              - 2KB True Dual-Port Unified Activation Buffer (UB)
//              - Weight Buffer (WB) for Stationary Weight Tiles
//              - Diagonal Wavefront Hardware Skew Unit
//              - 8x8 2D Weight-Stationary Systolic Array (64 PEs, 64 MACs/cycle)
//              - Output Column Hardware Deskew Unit
//              - Vector Processing Unit (VPU) with Bias, ReLU, GELU, and INT8 Quantization
// =============================================================================

`timescale 1ns / 1ps

module apex_tpu_top #(
    parameter ARRAY_SIZE     = 8,
    parameter AXI_ADDR_WIDTH = 32,
    parameter AXI_DATA_WIDTH = 64,
    parameter AXI_ID_WIDTH   = 4
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Instruction Interface (from Host / CPU)
    input  wire [31:0]                   instruction,
    input  wire                          instr_valid,
    output wire                          instr_ready,

    // TPU Status Signals
    output wire                          tpu_busy,
    output wire                          tpu_done,

    // External Bias Input Vector (ARRAY_SIZE * 32 bits = 256 bits)
    input  wire [ARRAY_SIZE*32-1:0]      ext_bias_flat,

    // -------------------------------------------------------------------------
    // AMBA AXI4 Master Interface (System Memory / DRAM)
    // -------------------------------------------------------------------------
    // Read Address Channel
    output wire [AXI_ID_WIDTH-1:0]       m_axi_arid,
    output wire [AXI_ADDR_WIDTH-1:0]     m_axi_araddr,
    output wire [7:0]                    m_axi_arlen,
    output wire [2:0]                    m_axi_arsize,
    output wire [1:0]                    m_axi_arburst,
    output wire                          m_axi_arvalid,
    input  wire                          m_axi_arready,

    // Read Data Channel
    input  wire [AXI_ID_WIDTH-1:0]       m_axi_rid,
    input  wire [AXI_DATA_WIDTH-1:0]     m_axi_rdata,
    input  wire [1:0]                    m_axi_rresp,
    input  wire                          m_axi_rlast,
    input  wire                          m_axi_rvalid,
    output wire                          m_axi_rready,

    // Write Address Channel
    output wire [AXI_ID_WIDTH-1:0]       m_axi_awid,
    output wire [AXI_ADDR_WIDTH-1:0]     m_axi_awaddr,
    output wire [7:0]                    m_axi_awlen,
    output wire [2:0]                    m_axi_awsize,
    output wire [1:0]                    m_axi_awburst,
    output wire                          m_axi_awvalid,
    input  wire                          m_axi_awready,

    // Write Data Channel
    output wire [AXI_DATA_WIDTH-1:0]     m_axi_wdata,
    output wire [(AXI_DATA_WIDTH/8)-1:0] m_axi_wstrb,
    output wire                          m_axi_wlast,
    output wire                          m_axi_wvalid,
    input  wire                          m_axi_wready,

    // Write Response Channel
    input  wire [AXI_ID_WIDTH-1:0]       m_axi_bid,
    input  wire [1:0]                    m_axi_bresp,
    input  wire                          m_axi_bvalid,
    output wire                          m_axi_bready
);

    // -------------------------------------------------------------------------
    // Internal Interconnect Wires
    // -------------------------------------------------------------------------
    // Controller <-> DMA
    wire        ctrl_dma_start;
    wire        ctrl_dma_dir;
    wire        ctrl_dma_target;
    wire [31:0] ctrl_dma_ext_addr;
    wire [7:0]  ctrl_dma_int_addr;
    wire [7:0]  ctrl_dma_length;
    wire        dma_ctrl_done;

    // DMA <-> Unified Buffer (Port A)
    wire        dma_ub_en;
    wire        dma_ub_we;
    wire [7:0]  dma_ub_addr;
    wire [63:0] dma_ub_wdata;
    wire [63:0] ub_dma_rdata;

    // DMA <-> Weight Buffer
    wire        dma_wb_we;
    wire [5:0]  dma_wb_addr;
    wire [63:0] dma_wb_wdata;

    // Controller <-> Weight Buffer
    wire [2:0]  ctrl_wb_tile_idx;
    wire        ctrl_wb_tile_load_en;
    wire [ARRAY_SIZE*ARRAY_SIZE*8-1:0] wb_sa_weights_flat;

    // Controller <-> Unified Buffer (Port B)
    wire        ctrl_ub_b_en;
    wire        ctrl_ub_b_we;
    wire [7:0]  ctrl_ub_b_addr;
    wire [63:0] ub_b_rdata;
    wire [63:0] vpu_ub_wdata;

    // Controller <-> Systolic Pipeline Controls
    wire        ctrl_sa_weight_load_en;
    wire [ARRAY_SIZE-1:0] ctrl_sa_weight_en_row;
    wire        ctrl_sa_compute_en;
    wire        ctrl_sa_clear_acc;
    wire        ctrl_skew_en;
    wire        ctrl_deskew_en;

    // Controller <-> VPU
    wire        ctrl_vpu_en;
    wire [1:0]  ctrl_vpu_act_func;
    wire        ctrl_vpu_bias_en;
    wire [3:0]  ctrl_vpu_quant_shift;
    wire        vpu_ctrl_valid_out;

    // Datapath Wires
    wire [ARRAY_SIZE*8-1:0]  skew_sa_act_flat;
    wire [ARRAY_SIZE*32-1:0] sa_deskew_sum_flat;
    wire [ARRAY_SIZE*8-1:0]  sa_act_out_unused;
    wire [ARRAY_SIZE*32-1:0] deskew_vpu_sum_flat;
    wire [ARRAY_SIZE*32-1:0] vpu_data_out_32_unused;

    // Top inputs to systolic array: initial partial sums (tied to 0)
    wire [ARRAY_SIZE*32-1:0] sa_sum_in_flat = {(ARRAY_SIZE*32){1'b0}};

    // -------------------------------------------------------------------------
    // Module Instantiations
    // -------------------------------------------------------------------------

    // 1. CISC TPU Instruction Controller
    tpu_controller #(
        .ARRAY_SIZE        (ARRAY_SIZE)
    ) u_controller (
        .clk               (clk),
        .rst_n             (rst_n),
        .instruction       (instruction),
        .instr_valid       (instr_valid),
        .instr_ready       (instr_ready),
        .tpu_busy          (tpu_busy),
        .tpu_done          (tpu_done),
        .dma_start         (ctrl_dma_start),
        .dma_dir           (ctrl_dma_dir),
        .dma_target        (ctrl_dma_target),
        .dma_ext_addr      (ctrl_dma_ext_addr),
        .dma_int_addr      (ctrl_dma_int_addr),
        .dma_length        (ctrl_dma_length),
        .dma_done          (dma_ctrl_done),
        .wb_tile_idx       (ctrl_wb_tile_idx),
        .wb_tile_load_en   (ctrl_wb_tile_load_en),
        .sa_weight_load_en (ctrl_sa_weight_load_en),
        .sa_weight_en_row  (ctrl_sa_weight_en_row),
        .sa_compute_en     (ctrl_sa_compute_en),
        .sa_clear_acc      (ctrl_sa_clear_acc),
        .ub_b_en           (ctrl_ub_b_en),
        .ub_b_we           (ctrl_ub_b_we),
        .ub_b_addr         (ctrl_ub_b_addr),
        .skew_en           (ctrl_skew_en),
        .deskew_en         (ctrl_deskew_en),
        .vpu_en            (ctrl_vpu_en),
        .vpu_act_func      (ctrl_vpu_act_func),
        .vpu_bias_en       (ctrl_vpu_bias_en),
        .vpu_quant_shift   (ctrl_vpu_quant_shift),
        .vpu_valid_out     (vpu_ctrl_valid_out)
    );

    // 2. AMBA AXI4 Master DMA Controller
    dma_axi_master #(
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .AXI_ID_WIDTH   (AXI_ID_WIDTH)
    ) u_dma_axi_master (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_start      (ctrl_dma_start),
        .cmd_dir        (ctrl_dma_dir),
        .cmd_target     (ctrl_dma_target),
        .cmd_ext_addr   (ctrl_dma_ext_addr),
        .cmd_int_addr   (ctrl_dma_int_addr),
        .cmd_length     (ctrl_dma_length),
        .cmd_done       (dma_ctrl_done),
        .cmd_busy       (),
        .ub_en          (dma_ub_en),
        .ub_we          (dma_ub_we),
        .ub_addr        (dma_ub_addr),
        .ub_wdata       (dma_ub_wdata),
        .ub_rdata       (ub_dma_rdata),
        .wb_we          (dma_wb_we),
        .wb_addr        (dma_wb_addr),
        .wb_wdata       (dma_wb_wdata),
        .m_axi_arid     (m_axi_arid),
        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_rid      (m_axi_rid),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready),
        .m_axi_awid     (m_axi_awid),
        .m_axi_awaddr   (m_axi_awaddr),
        .m_axi_awlen    (m_axi_awlen),
        .m_axi_awsize   (m_axi_awsize),
        .m_axi_awburst  (m_axi_awburst),
        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),
        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_bid      (m_axi_bid),
        .m_axi_bresp    (m_axi_bresp),
        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready)
    );

    // 3. True Dual-Port Unified Activation Buffer
    unified_buffer #(
        .DATA_WIDTH (64),
        .ADDR_WIDTH (8),
        .DEPTH      (256)
    ) u_unified_buffer (
        .clk        (clk),
        .rst_n      (rst_n),
        .a_en       (dma_ub_en),
        .a_we       (dma_ub_we),
        .a_addr     (dma_ub_addr),
        .a_wdata    (dma_ub_wdata),
        .a_rdata    (ub_dma_rdata),
        .b_en       (ctrl_ub_b_en),
        .b_we       (ctrl_ub_b_we),
        .b_addr     (ctrl_ub_b_addr),
        .b_wdata    (vpu_ub_wdata),
        .b_rdata    (ub_b_rdata)
    );

    // 4. Weight Buffer
    weight_buffer #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .ROW_WIDTH  (64),
        .DEPTH      (64),
        .ADDR_WIDTH (6)
    ) u_weight_buffer (
        .clk              (clk),
        .rst_n            (rst_n),
        .wr_en            (dma_wb_we),
        .wr_addr          (dma_wb_addr),
        .wr_data          (dma_wb_wdata),
        .tile_idx         (ctrl_wb_tile_idx),
        .tile_load_en     (ctrl_wb_tile_load_en),
        .weights_flat_out (wb_sa_weights_flat)
    );

    // 5. Input Skew Unit
    skew_unit #(
        .ARRAY_SIZE (ARRAY_SIZE)
    ) u_skew_unit (
        .clk        (clk),
        .rst_n      (rst_n),
        .enable     (ctrl_skew_en),
        .in_flat    (ub_b_rdata),
        .out_flat   (skew_sa_act_flat)
    );

    // 6. 2D Systolic Processing Grid (8x8 = 64 PEs)
    systolic_array #(
        .ARRAY_SIZE     (ARRAY_SIZE)
    ) u_systolic_array (
        .clk            (clk),
        .rst_n          (rst_n),
        .weight_load_en (ctrl_sa_weight_load_en),
        .weight_en_row  (ctrl_sa_weight_en_row),
        .compute_en     (ctrl_sa_compute_en),
        .clear_acc      (ctrl_sa_clear_acc),
        .weights_flat   (wb_sa_weights_flat),
        .act_in_flat    (skew_sa_act_flat),
        .sum_in_flat    (sa_sum_in_flat),
        .sum_out_flat   (sa_deskew_sum_flat),
        .act_out_flat   (sa_act_out_unused)
    );

    // 7. Output Deskew Unit
    deskew_unit #(
        .ARRAY_SIZE (ARRAY_SIZE)
    ) u_deskew_unit (
        .clk        (clk),
        .rst_n      (rst_n),
        .enable     (ctrl_deskew_en),
        .in_flat    (sa_deskew_sum_flat),
        .out_flat   (deskew_vpu_sum_flat)
    );

    // 8. Vector Processing Unit (VPU)
    vector_unit #(
        .ARRAY_SIZE       (ARRAY_SIZE)
    ) u_vector_unit (
        .clk              (clk),
        .rst_n            (rst_n),
        .enable           (ctrl_vpu_en),
        .act_func         (ctrl_vpu_act_func),
        .bias_en          (ctrl_vpu_bias_en),
        .quant_shift      (ctrl_vpu_quant_shift),
        .sum_in_flat      (deskew_vpu_sum_flat),
        .bias_in_flat     (ext_bias_flat),
        .data_out_32_flat (vpu_data_out_32_unused),
        .data_out_8_flat  (vpu_ub_wdata),
        .valid_out        (vpu_ctrl_valid_out)
    );

endmodule
