// =============================================================================
// File: tb_apex_tpu.v
// Module: tb_apex_tpu
// Description: Multi-Mode, Cycle-Accurate Hardware Verification Suite for Apex-TPU.
//              Verifies:
//              - Test Suite 1: Transformer GELU Mode (PWL GELU + Bias + Quant Shift=3)
//              - Test Suite 2: Deep Learning ReLU Mode (Strict max(0,x) + Bias + Quant Shift=2)
//              - Test Suite 3: Linear Pass-Through Mode (Linear GEMM + Bias + Quant Shift=4)
//              Features:
//              - Synthetic AXI4 Slave DRAM Memory Responder (4KB)
//              - CISC Instruction Dispatcher
//              - Comprehensive bit-exact comparison against NumPy golden references
// =============================================================================

`timescale 1ns / 1ps

module tb_apex_tpu;

    localparam ARRAY_SIZE     = 8;
    localparam AXI_ADDR_WIDTH = 32;
    localparam AXI_DATA_WIDTH = 64;
    localparam AXI_ID_WIDTH   = 4;
    localparam CLK_PERIOD     = 10; // 100 MHz clock

    // Clock and Reset
    reg clk;
    reg rst_n;

    // Instruction Interface
    reg  [31:0] instruction;
    reg         instr_valid;
    wire        instr_ready;
    wire        tpu_busy;
    wire        tpu_done;

    // External Bias Flat Vector: ARRAY_SIZE * 32 bits = 256 bits
    reg  [ARRAY_SIZE*32-1:0] ext_bias_flat;

    // AXI4 Bus Interconnect
    wire [AXI_ID_WIDTH-1:0]       axi_arid;
    wire [AXI_ADDR_WIDTH-1:0]     axi_araddr;
    wire [7:0]                    axi_arlen;
    wire [2:0]                    axi_arsize;
    wire [1:0]                    axi_arburst;
    wire                          axi_arvalid;
    reg                           axi_arready;

    reg  [AXI_ID_WIDTH-1:0]       axi_rid;
    reg  [AXI_DATA_WIDTH-1:0]     axi_rdata;
    reg  [1:0]                    axi_rresp;
    reg                           axi_rlast;
    reg                           axi_rvalid;
    wire                          axi_rready;

    wire [AXI_ID_WIDTH-1:0]       axi_awid;
    wire [AXI_ADDR_WIDTH-1:0]     axi_awaddr;
    wire [7:0]                    axi_awlen;
    wire [2:0]                    axi_awsize;
    wire [1:0]                    axi_awburst;
    wire                          axi_awvalid;
    reg                           axi_awready;

    wire [AXI_DATA_WIDTH-1:0]     axi_wdata;
    wire [(AXI_DATA_WIDTH/8)-1:0] axi_wstrb;
    wire                          axi_wlast;
    wire                          axi_wvalid;
    reg                           axi_wready;

    reg  [AXI_ID_WIDTH-1:0]       axi_bid;
    reg  [1:0]                    axi_bresp;
    reg                           axi_bvalid;
    wire                          axi_bready;

    // Instantiate Apex-TPU Top
    apex_tpu_top #(
        .ARRAY_SIZE     (ARRAY_SIZE),
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .AXI_ID_WIDTH   (AXI_ID_WIDTH)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .instruction    (instruction),
        .instr_valid    (instr_valid),
        .instr_ready    (instr_ready),
        .tpu_busy       (tpu_busy),
        .tpu_done       (tpu_done),
        .ext_bias_flat  (ext_bias_flat),
        .m_axi_arid     (axi_arid),
        .m_axi_araddr   (axi_araddr),
        .m_axi_arlen    (axi_arlen),
        .m_axi_arsize   (axi_arsize),
        .m_axi_arburst  (axi_arburst),
        .m_axi_arvalid  (axi_arvalid),
        .m_axi_arready  (axi_arready),
        .m_axi_rid      (axi_rid),
        .m_axi_rdata    (axi_rdata),
        .m_axi_rresp    (axi_rresp),
        .m_axi_rlast    (axi_rlast),
        .m_axi_rvalid   (axi_rvalid),
        .m_axi_rready   (axi_rready),
        .m_axi_awid     (axi_awid),
        .m_axi_awaddr   (axi_awaddr),
        .m_axi_awlen    (axi_awlen),
        .m_axi_awsize   (axi_awsize),
        .m_axi_awburst  (axi_awburst),
        .m_axi_awvalid  (axi_awvalid),
        .m_axi_awready  (axi_awready),
        .m_axi_wdata    (axi_wdata),
        .m_axi_wstrb    (axi_wstrb),
        .m_axi_wlast    (axi_wlast),
        .m_axi_wvalid   (axi_wvalid),
        .m_axi_wready   (axi_wready),
        .m_axi_bid      (axi_bid),
        .m_axi_bresp    (axi_bresp),
        .m_axi_bvalid   (axi_bvalid),
        .m_axi_bready   (axi_bready)
    );

    // -------------------------------------------------------------------------
    // Synthetic AXI4 Slave DRAM Memory Responder
    // -------------------------------------------------------------------------
    reg [63:0] dram_mem [0:511]; // 512 entries x 8 bytes = 4KB
    reg [63:0] expected_gelu [0:7];
    reg [63:0] expected_relu [0:7];
    reg [63:0] expected_pass [0:7];
    reg [31:0] bias_mem [0:7];

    reg [8:0]  rd_word_addr;
    reg [7:0]  rd_len_cnt;
    reg        rd_active;

    reg [8:0]  wr_word_addr;
    reg [7:0]  wr_len_cnt;
    reg        wr_active;

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // Simulation watchdog
    initial begin
        #250000;
        $display("[ERROR] Simulation Watchdog Timeout (250us)!");
        $finish;
    end

    // AXI Slave Read Engine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
            axi_rlast   <= 1'b0;
            axi_rdata   <= 64'd0;
            axi_rresp   <= 2'b00;
            rd_active   <= 1'b0;
            rd_len_cnt  <= 8'd0;
            rd_word_addr<= 9'd0;
        end else begin
            if (!rd_active) begin
                axi_arready <= 1'b1;
                if (axi_arvalid && axi_arready) begin
                    axi_arready  <= 1'b0;
                    rd_active    <= 1'b1;
                    rd_word_addr <= axi_araddr[11:3];
                    rd_len_cnt   <= axi_arlen;
                    axi_rvalid   <= 1'b1;
                    axi_rdata    <= dram_mem[axi_araddr[11:3]];
                    axi_rlast    <= (axi_arlen == 8'd0);
                end
            end else begin
                if (axi_rvalid && axi_rready) begin
                    if (rd_len_cnt == 8'd0) begin
                        axi_rvalid <= 1'b0;
                        axi_rlast  <= 1'b0;
                        rd_active  <= 1'b0;
                    end else begin
                        rd_word_addr <= rd_word_addr + 9'd1;
                        rd_len_cnt   <= rd_len_cnt - 8'd1;
                        axi_rdata    <= dram_mem[rd_word_addr + 9'd1];
                        axi_rlast    <= (rd_len_cnt == 8'd1);
                    end
                end
            end
        end
    end

    // AXI Slave Write Engine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
            axi_bvalid  <= 1'b0;
            axi_bresp   <= 2'b00;
            wr_active   <= 1'b0;
            wr_word_addr<= 9'd0;
            wr_len_cnt  <= 8'd0;
        end else begin
            if (axi_bvalid && axi_bready) begin
                axi_bvalid <= 1'b0;
            end

            if (!wr_active) begin
                axi_awready <= 1'b1;
                if (axi_awvalid && axi_awready) begin
                    axi_awready  <= 1'b0;
                    wr_active    <= 1'b1;
                    wr_word_addr <= axi_awaddr[11:3];
                    wr_len_cnt   <= axi_awlen;
                    axi_wready   <= 1'b1;
                end
            end else begin
                if (axi_wvalid && axi_wready) begin
                    dram_mem[wr_word_addr] <= axi_wdata;
                    wr_word_addr <= wr_word_addr + 9'd1;
                    if (axi_wlast || (wr_len_cnt == 8'd0)) begin
                        axi_wready <= 1'b0;
                        axi_bvalid <= 1'b1;
                        axi_bresp  <= 2'b00;
                        wr_active  <= 1'b0;
                    end else begin
                        wr_len_cnt <= wr_len_cnt - 8'd1;
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Helper Tasks
    // -------------------------------------------------------------------------
    task send_instr(input [31:0] cmd);
        begin
            while (!instr_ready) @(posedge clk);
            @(posedge clk);
            instruction <= cmd;
            instr_valid <= 1'b1;
            @(posedge clk);
            instruction <= 32'd0;
            instr_valid <= 1'b0;
        end
    endtask

    task wait_tpu_idle();
        begin
            @(posedge clk);
            while (tpu_busy) @(posedge clk);
        end
    endtask

    // -------------------------------------------------------------------------
    // Test Sequence
    // -------------------------------------------------------------------------
    integer r, c, total_errs, mode_errs;
    reg signed [7:0] act_byte, exp_byte;

    initial begin
        // Waveform dump
        $dumpfile("sim/apex_tpu.vcd");
        $dumpvars(0, tb_apex_tpu);

        // Load test vectors
        $readmemh("sim/dram_init.hex", dram_mem);
        $readmemh("sim/expected_gelu.hex", expected_gelu);
        $readmemh("sim/expected_relu.hex", expected_relu);
        $readmemh("sim/expected_pass.hex", expected_pass);
        $readmemh("sim/bias.hex", bias_mem);

        // Pack bias memory into flat bias vector
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
            ext_bias_flat[r*32 +: 32] = bias_mem[r];
        end

        // Initial reset
        rst_n       = 0;
        instruction = 32'd0;
        instr_valid = 0;
        total_errs  = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("==================================================================");
        $display("   APEX-TPU: COMPREHENSIVE MULTI-MODE VERIFICATION SUITE");
        $display("   Core: 8x8 Systolic Grid (64 PEs, 64 MACs/cycle, Weight-Stationary)");
        $display("==================================================================");

        // Preload Weights into Weight Buffer & Systolic Array PEs
        $display("[%0t ns] [STEP 1] Preloading 8x8 Stationary Weights from DRAM...", $time);
        send_instr(32'h10000008); // OP_LOAD_WEIGHT: DRAM Row 0 -> WB Row 0, len 8
        wait_tpu_idle();
        send_instr(32'h30000000); // OP_SET_WEIGHT: Tile 0 -> PEs
        wait_tpu_idle();

        // Load Input Activations into Unified Buffer
        $display("[%0t ns] [STEP 2] Preloading 8x8 Activation Matrix into Unified Buffer...", $time);
        send_instr(32'h20100008); // OP_LOAD_INPUT: DRAM Row 16 -> UB Row 0, len 8
        wait_tpu_idle();

        // =====================================================================
        // TEST SUITE 1: TRANSFORMER GELU MODE
        // =====================================================================
        $display("\n------------------------------------------------------------------");
        $display("TEST SUITE 1: TRANSFORMER GELU (Piecewise Linear GELU, Bias=ON, Shift=3)");
        $display("------------------------------------------------------------------");
        // OP_CONFIG_VPU: GELU (2'b10), Bias=1, Shift=3 => 32'h52130000
        send_instr(32'h52130000);
        wait_tpu_idle();
        // OP_MATMUL: UB rd=0, UB wr=64, K=8 => 32'h40004008
        send_instr(32'h40004008);
        wait_tpu_idle();
        // OP_STORE_OUT: UB row 64 -> DRAM row 32, len 8 => 32'h60204008
        send_instr(32'h60204008);
        wait_tpu_idle();

        mode_errs = 0;
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
            $write("Row %0d: Expected [ ", r);
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                exp_byte = expected_gelu[r][c*8 +: 8];
                $write("%4d ", exp_byte);
            end
            $write("] | Hardware [ ");
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                act_byte = dram_mem[32 + r][c*8 +: 8];
                exp_byte = expected_gelu[r][c*8 +: 8];
                $write("%4d ", act_byte);
                if (act_byte !== exp_byte) mode_errs = mode_errs + 1;
            end
            $display("]");
        end
        if (mode_errs == 0) $display(">>> [PASS] TRANSFORMER GELU: 64/64 ELEMENTS MATCHED (100%% BIT-EXACT) <<<");
        else $display(">>> [FAIL] TRANSFORMER GELU: %0d MISMATCHES! <<<", mode_errs);
        total_errs = total_errs + mode_errs;

        // =====================================================================
        // TEST SUITE 2: DEEP LEARNING RELU MODE
        // =====================================================================
        $display("\n------------------------------------------------------------------");
        $display("TEST SUITE 2: DEEP LEARNING RELU (Strict max(0,x), Bias=ON, Shift=2)");
        $display("------------------------------------------------------------------");
        // OP_CONFIG_VPU: ReLU (2'b01), Bias=1, Shift=2 => 32'h51120000
        send_instr(32'h51120000);
        wait_tpu_idle();
        // OP_MATMUL: UB rd=0, UB wr=72, K=8 => 32'h40004808
        send_instr(32'h40004808);
        wait_tpu_idle();
        // OP_STORE_OUT: UB row 72 -> DRAM row 40, len 8 => 32'h60284808
        send_instr(32'h60284808);
        wait_tpu_idle();

        mode_errs = 0;
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
            $write("Row %0d: Expected [ ", r);
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                exp_byte = expected_relu[r][c*8 +: 8];
                $write("%4d ", exp_byte);
            end
            $write("] | Hardware [ ");
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                act_byte = dram_mem[40 + r][c*8 +: 8];
                exp_byte = expected_relu[r][c*8 +: 8];
                $write("%4d ", act_byte);
                if (act_byte !== exp_byte) mode_errs = mode_errs + 1;
            end
            $display("]");
        end
        if (mode_errs == 0) $display(">>> [PASS] STANDARD RELU: 64/64 ELEMENTS MATCHED (100%% BIT-EXACT) <<<");
        else $display(">>> [FAIL] STANDARD RELU: %0d MISMATCHES! <<<", mode_errs);
        total_errs = total_errs + mode_errs;

        // =====================================================================
        // TEST SUITE 3: LINEAR PASS-THROUGH MODE
        // =====================================================================
        $display("\n------------------------------------------------------------------");
        $display("TEST SUITE 3: LINEAR PASS-THROUGH (Bypass, Bias=ON, Shift=4)");
        $display("------------------------------------------------------------------");
        // OP_CONFIG_VPU: Pass (2'b00), Bias=1, Shift=4 => 32'h50140000
        send_instr(32'h50140000);
        wait_tpu_idle();
        // OP_MATMUL: UB rd=0, UB wr=80, K=8 => 32'h40005008
        send_instr(32'h40005008);
        wait_tpu_idle();
        // OP_STORE_OUT: UB row 80 -> DRAM row 48, len 8 => 32'h60305008
        send_instr(32'h60305008);
        wait_tpu_idle();

        mode_errs = 0;
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
            $write("Row %0d: Expected [ ", r);
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                exp_byte = expected_pass[r][c*8 +: 8];
                $write("%4d ", exp_byte);
            end
            $write("] | Hardware [ ");
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                act_byte = dram_mem[48 + r][c*8 +: 8];
                exp_byte = expected_pass[r][c*8 +: 8];
                $write("%4d ", act_byte);
                if (act_byte !== exp_byte) mode_errs = mode_errs + 1;
            end
            $display("]");
        end
        if (mode_errs == 0) $display(">>> [PASS] LINEAR PASS-THROUGH: 64/64 ELEMENTS MATCHED (100%% BIT-EXACT) <<<");
        else $display(">>> [FAIL] LINEAR PASS-THROUGH: %0d MISMATCHES! <<<", mode_errs);
        total_errs = total_errs + mode_errs;

        // =====================================================================
        // SUMMARY
        // =====================================================================
        // OP_HALT
        send_instr(32'hF0000000);
        @(posedge clk);
        while (!tpu_done) @(posedge clk);

        $display("\n==================================================================");
        if (total_errs == 0) begin
            $display(">>> ALL 3 TEST SUITES PASSED: 192/192 MATRIX ELEMENTS VERIFIED! <<<");
            $display(">>> ZERO DEFECTS DETECTED ACROSS ALL DL ACTIVATION PIPELINES! <<<");
        end else begin
            $display(">>> TOTAL DEFECTS DETECTED: %0d <<<", total_errs);
        end
        $display("==================================================================");

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule
