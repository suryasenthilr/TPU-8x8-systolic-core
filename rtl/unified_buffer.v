// =============================================================================
// File: unified_buffer.v
// Module: unified_buffer
// Description: Unified Activation Buffer (UB) for Apex-TPU.
//              True Dual-Port synchronous SRAM scratchpad.
//              - Port A: Connected to AXI4 DMA interface (host/memory transfers).
//              - Port B: Connected to Systolic Array read / VPU writeback.
//              Data width is ARRAY_SIZE * 8 bits (e.g. 64 bits for 8x8 array).
// =============================================================================

`timescale 1ns / 1ps

module unified_buffer #(
    parameter DATA_WIDTH = 64,   // ARRAY_SIZE * 8 bits
    parameter ADDR_WIDTH = 8,    // 256 entries = 2048 bytes
    parameter DEPTH      = 256
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Port A: DMA / System Memory Interface
    input  wire                  a_en,
    input  wire                  a_we,
    input  wire [ADDR_WIDTH-1:0] a_addr,
    input  wire [DATA_WIDTH-1:0] a_wdata,
    output reg  [DATA_WIDTH-1:0] a_rdata,

    // Port B: TPU Compute Pipeline (Skew in / VPU out)
    input  wire                  b_en,
    input  wire                  b_we,
    input  wire [ADDR_WIDTH-1:0] b_addr,
    input  wire [DATA_WIDTH-1:0] b_wdata,
    output reg  [DATA_WIDTH-1:0] b_rdata
);

    // Memory array
    reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];

    // Port A synchronous read/write
    always @(posedge clk) begin
        if (a_en) begin
            if (a_we) begin
                ram[a_addr] <= a_wdata;
            end
            a_rdata <= ram[a_addr];
        end
    end

    // Port B synchronous read/write
    always @(posedge clk) begin
        if (b_en) begin
            if (b_we) begin
                ram[b_addr] <= b_wdata;
            end
            b_rdata <= ram[b_addr];
        end
    end

endmodule
