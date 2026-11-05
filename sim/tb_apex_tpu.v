// =============================================================================
// File: tb_apex_tpu.v
// Module: tb_apex_tpu
// Description: Cycle-accurate, self-checking hardware testbench for Apex-TPU.
//              Features:
//              - Synthetic AXI4 Slave DRAM Memory Responder
//              - CISC TPU Instruction Dispatcher
//              - End-to-End Verification of INT8 Systolic GEMM + Bias + PWL GELU
//              - Bit-exact comparison against Golden NumPy Model
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
    reg [63:0] expected_out [0:7];
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
        #100000;
        $display("[ERROR] Simulation Watchdog Timeout (100us)!");
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
            // Ready to accept new AR if not actively bursting
            if (!rd_active) begin
                axi_arready <= 1'b1;
                if (axi_arvalid && axi_arready) begin
                    axi_arready  <= 1'b0;
                    rd_active    <= 1'b1;
                    rd_word_addr <= axi_araddr[11:3]; // Word address (64-bit word)
                    rd_len_cnt   <= axi_arlen;
                    axi_rvalid   <= 1'b1;
                    axi_rdata    <= dram_mem[axi_araddr[11:3]];
                    axi_rlast    <= (axi_arlen == 8'd0);
                end
            end else begin
                // Bursting read beats
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
            // Defaults
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
    // Helper Task: Send Instruction
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

    // Helper Task: Wait for TPU idle
    task wait_tpu_idle();
        begin
            @(posedge clk);
            while (tpu_busy) @(posedge clk);
        end
    endtask

    // -------------------------------------------------------------------------
    // Test Sequence
    // -------------------------------------------------------------------------
    integer r, c, err_cnt, cycle_start, cycle_end, total_cycles;
    reg signed [7:0] act_byte, exp_byte;

    initial begin
        // Waveform dump
        $dumpfile("sim/apex_tpu.vcd");
        $dumpvars(0, tb_apex_tpu);

        // Load test vectors
        $readmemh("sim/dram_init.hex", dram_mem);
        $readmemh("sim/expected_gelu.hex", expected_out);
        $readmemh("sim/bias.hex", bias_mem);

        // Pack bias memory into flat bias vector
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
            ext_bias_flat[r*32 +: 32] = bias_mem[r];
        end

        // Initial reset
        rst_n       = 0;
        instruction = 32'd0;
        instr_valid = 0;
        err_cnt     = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("==================================================================");
        $display("   APEX-TPU: 8x8 SYSTOLIC ARRAY DL ACCELERATOR VERIFICATION");
        $display("==================================================================");

        cycle_start = $time / CLK_PERIOD;

        // 1. OP_LOAD_WEIGHT: DRAM Row 0 -> Weight Buffer Row 0, length 8
        // Opcode=0x1, ext_row=0x00, int_row=0x00, len=0x08 => 32'h10000008
        $display("[%0t ns] [DISPATCH] OP_LOAD_WEIGHT: Preloading 8x8 Weight Matrix from DRAM to WB...", $time);
        send_instr(32'h10000008);
        wait_tpu_idle();
        $display("[%0t ns] [STATUS] Weight DMA transfer complete.", $time);

        // 2. OP_SET_WEIGHT: Tile 0 -> Systolic Array PEs
        // Opcode=0x3, tile_idx=0 => 32'h30000000
        $display("[%0t ns] [DISPATCH] OP_SET_WEIGHT: Preloading Stationary Weights into 64 PEs...", $time);
        send_instr(32'h30000000);
        wait_tpu_idle();
        $display("[%0t ns] [STATUS] Weights stationary in PE registers.", $time);

        // 3. OP_LOAD_INPUT: DRAM Row 16 -> Unified Buffer Row 0, length 8
        // Opcode=0x2, ext_row=0x10, int_row=0x00, len=0x08 => 32'h20100008
        $display("[%0t ns] [DISPATCH] OP_LOAD_INPUT: Streaming 8x8 Activation Matrix from DRAM to UB...", $time);
        send_instr(32'h20100008);
        wait_tpu_idle();
        $display("[%0t ns] [STATUS] Activations loaded into Unified Buffer.", $time);

        // 4. OP_CONFIG_VPU: Act=GELU (2'b10), Bias=1, Quant_Shift=3 (4'd3)
        // Opcode=0x5, act[25:24]=2'b10, bias[20]=1'b1, shift[19:16]=4'h3
        // 32'b0101_0010_0001_0011_0000_0000_0000_0000 => 32'h52130000
        $display("[%0t ns] [DISPATCH] OP_CONFIG_VPU: Config VPU (GELU, Bias=ON, Shift=3)...", $time);
        send_instr(32'h52130000);
        wait_tpu_idle();

        // 5. OP_MATMUL: UB rd_base=0x00, UB wr_base=0x40 (row 64), K=8
        // Opcode=0x4, rd_base=0x00, wr_base=0x40, K=0x08 => 32'h40004008
        $display("[%0t ns] [DISPATCH] OP_MATMUL: Executing 2D Systolic Wavefront Matrix Multiplication...", $time);
        send_instr(32'h40004008);
        wait_tpu_idle();
        $display("[%0t ns] [STATUS] GEMM systolic wavefront completed and written back to UB.", $time);

        // 6. OP_STORE_OUT: UB Row 64 -> DRAM Row 32 (0x20), length 8
        // Opcode=0x6, ext_row=0x20, int_row=0x40, len=0x08 => 32'h60204008
        $display("[%0t ns] [DISPATCH] OP_STORE_OUT: DMA Transfer of Quantized Results to DRAM...", $time);
        send_instr(32'h60204008);
        wait_tpu_idle();
        $display("[%0t ns] [STATUS] Results stored in system DRAM.", $time);

        // 7. OP_HALT
        // Opcode=0xF => 32'hF0000000
        send_instr(32'hF0000000);
        @(posedge clk);
        while (!tpu_done) @(posedge clk);

        cycle_end = $time / CLK_PERIOD;
        total_cycles = cycle_end - cycle_start;

        $display("------------------------------------------------------------------");
        $display("[BENCHMARK] Total Execution Latency: %0d clock cycles (%0.2f ns @ 100MHz)", total_cycles, total_cycles * CLK_PERIOD);
        $display("[BENCHMARK] Peak Compute Throughput: 64 MACs/cycle (12.8 GOPs/s @ 100MHz)");

        // ---------------------------------------------------------------------
        // Check results against Golden Model
        // ---------------------------------------------------------------------
        $display("------------------------------------------------------------------");
        $display("VERIFYING RESULTS AGAINST GOLDEN PYTHON NUMPY MODEL...");

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
            $write("Row %0d: Expected [ ", r);
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                exp_byte = expected_out[r][c*8 +: 8];
                $write("%4d ", exp_byte);
            end
            $write("] | Hardware [ ");
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                act_byte = dram_mem[32 + r][c*8 +: 8];
                exp_byte = expected_out[r][c*8 +: 8];
                $write("%4d ", act_byte);
                if (act_byte !== exp_byte) begin
                    err_cnt = err_cnt + 1;
                end
            end
            $display("]");
        end

        $display("==================================================================");
        if (err_cnt == 0) begin
            $display(">>> TEST PASSED: ALL 64 MATRIX ELEMENTS MATCHED WITH 100%% BIT-EXACT ACCURACY! <<<");
        end else begin
            $display(">>> TEST FAILED: %0d MISMATCHES DETECTED! <<<", err_cnt);
        end
        $display("==================================================================");

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule
