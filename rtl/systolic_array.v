// =============================================================================
// File: systolic_array.v
// Module: systolic_array
// Description: 2D Systolic Processing Grid (8x8 = 64 Processing Elements).
//              Implements Weight-Stationary spatial execution where activations
//              stream horizontally across rows and partial sums accumulate
//              vertically across columns.
// =============================================================================

`timescale 1ns / 1ps

module systolic_array #(
    parameter ARRAY_SIZE = 8
)(
    input  wire                               clk,
    input  wire                               rst_n,

    // Array Global Controls
    input  wire                               weight_load_en,
    input  wire [ARRAY_SIZE-1:0]              weight_en_row,   // Row-selective weight preloading
    input  wire                               compute_en,
    input  wire                               clear_acc,

    // Flattened Preload Weights: ARRAY_SIZE * ARRAY_SIZE * 8 bits
    input  wire [ARRAY_SIZE*ARRAY_SIZE*8-1:0] weights_flat,

    // Left Input: Activations for rows 0..(ARRAY_SIZE-1)
    input  wire [ARRAY_SIZE*8-1:0]            act_in_flat,

    // Top Input: Initial partial sums for columns 0..(ARRAY_SIZE-1) (typically 0)
    input  wire [ARRAY_SIZE*32-1:0]           sum_in_flat,

    // Bottom Output: Final accumulated sums from columns 0..(ARRAY_SIZE-1)
    output wire [ARRAY_SIZE*32-1:0]           sum_out_flat,

    // Right Output: Activations passing out of right edge of rows
    output wire [ARRAY_SIZE*8-1:0]            act_out_flat
);

    // Internal 2D interconnect buses
    wire [7:0]  act_bus [ARRAY_SIZE-1:0][ARRAY_SIZE:0];
    wire [31:0] sum_bus [ARRAY_SIZE:0][ARRAY_SIZE-1:0];

    genvar r, c;
    generate
        // Connect left inputs and right outputs
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin : gen_left_right
            assign act_bus[r][0] = act_in_flat[r*8 +: 8];
            assign act_out_flat[r*8 +: 8] = act_bus[r][ARRAY_SIZE];
        end

        // Connect top inputs and bottom outputs
        for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : gen_top_bottom
            assign sum_bus[0][c] = sum_in_flat[c*32 +: 32];
            assign sum_out_flat[c*32 +: 32] = sum_bus[ARRAY_SIZE][c];
        end

        // 2D Processing Element Grid
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin : gen_rows
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : gen_cols
                // Extract weight for this specific PE: weight[r][c]
                wire [7:0] pe_weight = weights_flat[(r*ARRAY_SIZE + c)*8 +: 8];
                wire       pe_w_en   = weight_load_en && weight_en_row[r];

                pe u_pe (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .weight_en  (pe_w_en),
                    .compute_en (compute_en),
                    .clear_acc  (clear_acc),
                    .weight_in  (pe_weight),
                    .act_in     (act_bus[r][c]),
                    .sum_in     (sum_bus[r][c]),
                    .act_out    (act_bus[r][c+1]),
                    .sum_out    (sum_bus[r+1][c])
                );
            end
        end
    endgenerate

endmodule
