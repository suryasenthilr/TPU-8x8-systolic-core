// =============================================================================
// File: weight_buffer.v
// Module: weight_buffer
// Description: Weight Buffer (WB) for Apex-TPU.
//              Holds preloaded tiled weight matrices from external memory.
//              Supports writing row-by-row from DMA and presenting complete
//              (ARRAY_SIZE x ARRAY_SIZE) 512-bit stationary weight tiles to the
//              systolic array.
// =============================================================================

`timescale 1ns / 1ps

module weight_buffer #(
    parameter ARRAY_SIZE = 8,
    parameter ROW_WIDTH  = 64,   // ARRAY_SIZE * 8 bits
    parameter DEPTH      = 64,   // 64 rows = 8 full (8x8) weight tiles
    parameter ADDR_WIDTH = 6
)(
    input  wire                               clk,
    input  wire                               rst_n,

    // Host / DMA Write Interface
    input  wire                               wr_en,
    input  wire [ADDR_WIDTH-1:0]              wr_addr,
    input  wire [ROW_WIDTH-1:0]               wr_data,

    // TPU Core Read Interface
    input  wire [ADDR_WIDTH-1-3:0]            tile_idx,       // Selects 1 of 8 tiles (each tile is 8 rows)
    input  wire                               tile_load_en,   // Strobe to latch tile to output registers

    // Direct parallel weights to systolic array: 8x8x8 = 512 bits
    output reg  [ARRAY_SIZE*ARRAY_SIZE*8-1:0] weights_flat_out
);

    // Weight Storage SRAM: 64 rows of 64 bits
    reg [ROW_WIDTH-1:0] weight_ram [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en) begin
            weight_ram[wr_addr] <= wr_data;
        end
    end

    // Tile latching: Assembles 8 rows from the selected tile into 512-bit weights_flat_out
    wire [ADDR_WIDTH-1:0] tile_base_addr = {tile_idx, 3'b000};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weights_flat_out <= {(ARRAY_SIZE*ARRAY_SIZE*8){1'b0}};
        end else if (tile_load_en) begin
            weights_flat_out[0*64 +: 64] <= weight_ram[tile_base_addr + 0];
            weights_flat_out[1*64 +: 64] <= weight_ram[tile_base_addr + 1];
            weights_flat_out[2*64 +: 64] <= weight_ram[tile_base_addr + 2];
            weights_flat_out[3*64 +: 64] <= weight_ram[tile_base_addr + 3];
            weights_flat_out[4*64 +: 64] <= weight_ram[tile_base_addr + 4];
            weights_flat_out[5*64 +: 64] <= weight_ram[tile_base_addr + 5];
            weights_flat_out[6*64 +: 64] <= weight_ram[tile_base_addr + 6];
            weights_flat_out[7*64 +: 64] <= weight_ram[tile_base_addr + 7];
        end
    end

endmodule
