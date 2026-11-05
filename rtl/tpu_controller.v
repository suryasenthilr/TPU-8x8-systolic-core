// =============================================================================
// File: tpu_controller.v
// Module: tpu_controller
// Description: CISC Matrix Instruction Decoder and Sequencer for Apex-TPU.
//              Decodes instructions:
//              - OP_NOP         (0x0)
//              - OP_LOAD_WEIGHT (0x1): DRAM -> Weight Buffer via DMA
//              - OP_LOAD_INPUT  (0x2): DRAM -> Unified Buffer via DMA
//              - OP_SET_WEIGHT  (0x3): Weight Buffer -> Systolic Array PEs
//              - OP_MATMUL      (0x4): Stream UB -> Skew -> Array -> Deskew -> VPU -> UB
//              - OP_CONFIG_VPU  (0x5): Configure Bias, ReLU/GELU, Quant Shift
//              - OP_STORE_OUT   (0x6): Unified Buffer -> DRAM via DMA
//              - OP_HALT        (0xF): Execution Complete
// =============================================================================

`timescale 1ns / 1ps

module tpu_controller #(
    parameter ARRAY_SIZE = 8
)(
    input  wire         clk,
    input  wire         rst_n,

    // Instruction Interface
    input  wire [31:0]  instruction,
    input  wire         instr_valid,
    output reg          instr_ready,

    // Status
    output reg          tpu_busy,
    output reg          tpu_done,

    // DMA Control Interface
    output reg          dma_start,
    output reg          dma_dir,
    output reg          dma_target,
    output reg  [31:0]  dma_ext_addr,
    output reg  [7:0]   dma_int_addr,
    output reg  [7:0]   dma_length,
    input  wire         dma_done,

    // Weight Buffer Control
    output reg  [2:0]   wb_tile_idx,
    output reg          wb_tile_load_en,

    // Systolic Array Controls
    output reg          sa_weight_load_en,
    output reg [ARRAY_SIZE-1:0] sa_weight_en_row,
    output reg          sa_compute_en,
    output reg          sa_clear_acc,

    // Unified Buffer Port B (Compute Stream)
    output reg          ub_b_en,
    output reg          ub_b_we,
    output reg  [7:0]   ub_b_addr,

    // Skew & Deskew Unit Controls
    output reg          skew_en,
    output reg          deskew_en,

    // Vector Unit Controls
    output reg          vpu_en,
    output reg  [1:0]   vpu_act_func,
    output reg          vpu_bias_en,
    output reg  [3:0]   vpu_quant_shift,
    input  wire         vpu_valid_out
);

    // Instruction Opcodes
    localparam OP_NOP         = 4'h0;
    localparam OP_LOAD_WEIGHT = 4'h1;
    localparam OP_LOAD_INPUT  = 4'h2;
    localparam OP_SET_WEIGHT  = 4'h3;
    localparam OP_MATMUL      = 4'h4;
    localparam OP_CONFIG_VPU  = 4'h5;
    localparam OP_STORE_OUT   = 4'h6;
    localparam OP_HALT        = 4'hF;

    // FSM States
    localparam S_IDLE          = 4'd0;
    localparam S_DECODE        = 4'd1;
    localparam S_DMA_WAIT      = 4'd2;
    localparam S_SET_WEIGHT_0  = 4'd3;
    localparam S_SET_WEIGHT_1  = 4'd4;
    localparam S_MATMUL_FEED   = 4'd5;
    localparam S_MATMUL_DRAIN  = 4'd6;
    localparam S_HALT          = 4'd7;

    reg [3:0]  state;
    reg [31:0] instr_reg;
    reg [7:0]  feed_cnt;
    reg [7:0]  matmul_cyc;
    reg [7:0]  k_dim;
    reg [7:0]  ub_rd_base;
    reg [7:0]  ub_wr_base;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            instr_reg         <= 32'd0;
            instr_ready       <= 1'b1;
            tpu_busy          <= 1'b0;
            tpu_done          <= 1'b0;

            dma_start         <= 1'b0;
            dma_dir           <= 1'b0;
            dma_target        <= 1'b0;
            dma_ext_addr      <= 32'd0;
            dma_int_addr      <= 8'd0;
            dma_length        <= 8'd0;

            wb_tile_idx       <= 3'd0;
            wb_tile_load_en   <= 1'b0;

            sa_weight_load_en <= 1'b0;
            sa_weight_en_row  <= {ARRAY_SIZE{1'b0}};
            sa_compute_en     <= 1'b0;
            sa_clear_acc      <= 1'b0;

            ub_b_en           <= 1'b0;
            ub_b_we           <= 1'b0;
            ub_b_addr         <= 8'd0;

            skew_en           <= 1'b0;
            deskew_en         <= 1'b0;

            vpu_en            <= 1'b0;
            vpu_act_func      <= 2'b00;
            vpu_bias_en       <= 1'b0;
            vpu_quant_shift   <= 4'd0;

            feed_cnt          <= 8'd0;
            matmul_cyc        <= 8'd0;
            k_dim             <= 8'd0;
            ub_rd_base        <= 8'd0;
            ub_wr_base        <= 8'd0;
        end else begin
            // Single-cycle strobes default off
            dma_start         <= 1'b0;
            wb_tile_load_en   <= 1'b0;
            sa_weight_load_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    instr_ready <= 1'b1;
                    tpu_busy    <= 1'b0;
                    if (instr_valid) begin
                        instr_ready <= 1'b0;
                        tpu_busy    <= 1'b1;
                        instr_reg   <= instruction;
                        state       <= S_DECODE;
                    end
                end

                S_DECODE: begin
                    case (instr_reg[31:28])
                        OP_NOP: begin
                            state <= S_IDLE;
                        end

                        OP_LOAD_WEIGHT: begin
                            // [23:16] ext row, [15:8] int row, [7:0] len
                            dma_start    <= 1'b1;
                            dma_dir      <= 1'b0; // Read from DRAM
                            dma_target   <= 1'b1; // To Weight Buffer
                            dma_ext_addr <= {21'd0, instr_reg[23:16], 3'b000}; // Byte addr = row * 8
                            dma_int_addr <= instr_reg[15:8];
                            dma_length   <= (instr_reg[7:0] == 8'd0) ? 8'd8 : instr_reg[7:0];
                            state        <= S_DMA_WAIT;
                        end

                        OP_LOAD_INPUT: begin
                            // [23:16] ext row, [15:8] int row, [7:0] len
                            dma_start    <= 1'b1;
                            dma_dir      <= 1'b0; // Read from DRAM
                            dma_target   <= 1'b0; // To Unified Buffer
                            dma_ext_addr <= {21'd0, instr_reg[23:16], 3'b000}; // Byte addr = row * 8
                            dma_int_addr <= instr_reg[15:8];
                            dma_length   <= (instr_reg[7:0] == 8'd0) ? 8'd8 : instr_reg[7:0];
                            state        <= S_DMA_WAIT;
                        end

                        OP_SET_WEIGHT: begin
                            wb_tile_idx     <= instr_reg[26:24];
                            wb_tile_load_en <= 1'b1;
                            state           <= S_SET_WEIGHT_0;
                        end

                        OP_CONFIG_VPU: begin
                            vpu_act_func    <= instr_reg[25:24];
                            vpu_bias_en     <= instr_reg[20];
                            vpu_quant_shift <= instr_reg[19:16];
                            state           <= S_IDLE;
                        end

                        OP_MATMUL: begin
                            ub_rd_base        <= instr_reg[23:16];
                            ub_wr_base        <= instr_reg[15:8];
                            k_dim             <= (instr_reg[7:0] == 8'd0) ? 8'd8 : instr_reg[7:0];
                            feed_cnt          <= 8'd0;
                            matmul_cyc        <= 8'd0;
                            
                            // Initialize systolic pipeline controls
                            sa_compute_en     <= 1'b1;
                            skew_en           <= 1'b1;
                            deskew_en         <= 1'b1;
                            vpu_en            <= 1'b1;
                            sa_clear_acc      <= 1'b0;

                            // Read row 0 of activations
                            ub_b_en           <= 1'b1;
                            ub_b_we           <= 1'b0;
                            ub_b_addr         <= instr_reg[23:16];

                            state             <= S_MATMUL_FEED;
                        end

                        OP_STORE_OUT: begin
                            dma_start    <= 1'b1;
                            dma_dir      <= 1'b1; // Write to DRAM
                            dma_target   <= 1'b0; // From Unified Buffer
                            dma_ext_addr <= {21'd0, instr_reg[23:16], 3'b000}; // Byte addr = row * 8
                            dma_int_addr <= instr_reg[15:8];
                            dma_length   <= (instr_reg[7:0] == 8'd0) ? 8'd8 : instr_reg[7:0];
                            state        <= S_DMA_WAIT;
                        end

                        OP_HALT: begin
                            tpu_done <= 1'b1;
                            tpu_busy <= 1'b0;
                            state    <= S_HALT;
                        end

                        default: state <= S_IDLE;
                    endcase
                end

                S_DMA_WAIT: begin
                    if (dma_done) begin
                        state <= S_IDLE;
                    end
                end

                S_SET_WEIGHT_0: begin
                    // One cycle for weight buffer SRAM to register output
                    sa_weight_load_en <= 1'b1;
                    sa_weight_en_row  <= {ARRAY_SIZE{1'b1}};
                    state             <= S_SET_WEIGHT_1;
                end

                S_SET_WEIGHT_1: begin
                    // Latch completed
                    sa_weight_load_en <= 1'b0;
                    sa_weight_en_row  <= {ARRAY_SIZE{1'b0}};
                    state             <= S_IDLE;
                end

                // Feed phase: Stream K rows of activations into Skew Unit
                S_MATMUL_FEED: begin
                    matmul_cyc <= matmul_cyc + 8'd1;
                    feed_cnt   <= feed_cnt + 8'd1;

                    if (feed_cnt + 8'd1 < k_dim) begin
                        ub_b_en   <= 1'b1;
                        ub_b_we   <= 1'b0;
                        ub_b_addr <= ub_rd_base + feed_cnt + 8'd1;
                    end else begin
                        // All activation rows read; turn off UB read
                        ub_b_en <= 1'b0;
                        ub_b_we <= 1'b0;
                        state   <= S_MATMUL_DRAIN;
                    end
                end

                // Drain phase: Allow remaining systolic wavefront to flush through deskew & VPU
                S_MATMUL_DRAIN: begin
                    matmul_cyc <= matmul_cyc + 8'd1;

                    if (matmul_cyc >= 8'd16 && matmul_cyc < 8'd16 + k_dim) begin
                        // Active writeback window: exactly K cycles
                        ub_b_en   <= 1'b1;
                        ub_b_we   <= 1'b1;
                        ub_b_addr <= ub_wr_base + (matmul_cyc - 8'd16);
                    end else begin
                        ub_b_en   <= 1'b0;
                        ub_b_we   <= 1'b0;
                    end

                    if (matmul_cyc >= 8'd16 + k_dim) begin
                        ub_b_en       <= 1'b0;
                        ub_b_we       <= 1'b0;
                        sa_compute_en <= 1'b0;
                        skew_en       <= 1'b0;
                        deskew_en     <= 1'b0;
                        vpu_en        <= 1'b0;
                        state         <= S_IDLE;
                    end
                end

                S_HALT: begin
                    // Remains halted until reset
                    tpu_done <= 1'b1;
                    tpu_busy <= 1'b0;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
