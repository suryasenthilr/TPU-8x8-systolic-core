// =============================================================================
// File: skew_unit.v
// Module: skew_unit
// Description: Systolic Input Skewing Unit.
//              Delays activation inputs along rows to form a diagonal wavefront
//              for the 2D Weight-Stationary systolic array.
//              Row r is delayed by r clock cycles (r = 0 .. ARRAY_SIZE-1).
// =============================================================================

`timescale 1ns / 1ps

module skew_unit #(
    parameter ARRAY_SIZE = 8
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      enable,

    // Parallel activation inputs: ARRAY_SIZE elements (8 bits each)
    input  wire [ARRAY_SIZE*8-1:0]   in_flat,

    // Diagonal wavefront skewed activation outputs
    output wire [ARRAY_SIZE*8-1:0]   out_flat
);

    genvar r, k;
    generate
        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin : gen_skew_row
            if (r == 0) begin : gen_row0
                // Row 0 has 0 delay cycles (direct connection)
                assign out_flat[7:0] = in_flat[7:0];
            end else begin : gen_row_delay
                // Row r requires r delay stages
                reg [7:0] delay_pipe [r-1:0];
                integer j;

                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (j = 0; j < r; j = j + 1) begin
                            delay_pipe[j] <= 8'd0;
                        end
                    end else if (enable) begin
                        delay_pipe[0] <= in_flat[r*8 +: 8];
                        for (j = 1; j < r; j = j + 1) begin
                            delay_pipe[j] <= delay_pipe[j-1];
                        end
                    end
                end

                assign out_flat[r*8 +: 8] = delay_pipe[r-1];
            end
        end
    endgenerate

endmodule
