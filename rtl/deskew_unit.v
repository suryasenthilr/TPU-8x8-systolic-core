// =============================================================================
// File: deskew_unit.v
// Module: deskew_unit
// Description: Systolic Output Deskewing Unit.
//              Realigns vertically emerging partial sums from systolic array columns
//              into parallel row vectors.
//              Column c emerges at (t + ARRAY_SIZE + c), so column c is delayed by
//              (ARRAY_SIZE - 1 - c) clock cycles (c = 0 .. ARRAY_SIZE-1).
// =============================================================================

`timescale 1ns / 1ps

module deskew_unit #(
    parameter ARRAY_SIZE = 8
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       enable,

    // Raw bottom outputs from systolic array: ARRAY_SIZE elements (32 bits each)
    input  wire [ARRAY_SIZE*32-1:0]   in_flat,

    // Synchronously realigned parallel outputs: ARRAY_SIZE elements (32 bits each)
    output wire [ARRAY_SIZE*32-1:0]   out_flat
);

    genvar c;
    generate
        for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : gen_deskew_col
            localparam DELAY = (ARRAY_SIZE - 1) - c;

            if (DELAY == 0) begin : gen_col_nodelay
                // Column (ARRAY_SIZE-1) has 0 delay cycles (direct connection)
                assign out_flat[c*32 +: 32] = in_flat[c*32 +: 32];
            end else begin : gen_col_delay
                // Column c requires DELAY stages
                reg [31:0] delay_pipe [DELAY-1:0];
                integer j;

                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (j = 0; j < DELAY; j = j + 1) begin
                            delay_pipe[j] <= 32'd0;
                        end
                    end else if (enable) begin
                        delay_pipe[0] <= in_flat[c*32 +: 32];
                        for (j = 1; j < DELAY; j = j + 1) begin
                            delay_pipe[j] <= delay_pipe[j-1];
                        end
                    end
                end

                assign out_flat[c*32 +: 32] = delay_pipe[DELAY-1];
            end
        end
    endgenerate

endmodule
