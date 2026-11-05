// =============================================================================
// File: pe.v
// Module: pe
// Description: Processing Element (PE) for 2D Systolic Tensor Processing Unit.
//              Implements Weight-Stationary (WS) dataflow with local weight storage,
//              signed 8-bit multiplication, 32-bit partial sum accumulation,
//              and horizontal activation propagation.
// =============================================================================

`timescale 1ns / 1ps

module pe (
    input  wire        clk,
    input  wire        rst_n,

    // Control Signals
    input  wire        weight_en,       // Preload weight into local register
    input  wire        compute_en,      // Step systolic wavefront execution
    input  wire        clear_acc,       // Clear accumulator / pass-through mode

    // Data Inputs
    input  wire [7:0]  weight_in,       // Signed 8-bit weight input for preloading
    input  wire [7:0]  act_in,          // Signed 8-bit activation streamed from left
    input  wire [31:0] sum_in,          // Signed 32-bit partial sum streamed from top

    // Data Outputs
    output reg  [7:0]  act_out,         // Forward activation to PE on right
    output reg  [31:0] sum_out          // Forward accumulated sum to PE below
);

    // Local Stationary Weight Storage
    reg signed [7:0] weight_reg;

    // Signed arithmetic casts
    wire signed [7:0]  s_act    = act_in;
    wire signed [7:0]  s_weight = weight_reg;
    wire signed [15:0] s_prod   = s_act * s_weight;
    wire signed [31:0] s_sum_in = sum_in;

    // Partial sum calculation
    wire signed [31:0] next_sum = clear_acc ? {{16{s_prod[15]}}, s_prod} 
                                            : (s_sum_in + {{16{s_prod[15]}}, s_prod});

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 8'sd0;
            act_out    <= 8'd0;
            sum_out    <= 32'd0;
        end else begin
            // 1. Weight Preload Phase
            if (weight_en) begin
                weight_reg <= weight_in;
            end

            // 2. Systolic Wavefront Execution Phase
            if (compute_en) begin
                act_out <= act_in;
                sum_out <= next_sum;
            end
        end
    end

endmodule
