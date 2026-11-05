// =============================================================================
// File: vector_unit.v
// Module: vector_unit
// Description: Vector Processing Unit (VPU) for Apex-TPU.
//              Processes ARRAY_SIZE parallel 32-bit channels:
//              1. Signed Bias Addition
//              2. Activation Function:
//                 - 2'b00: BYPASS (Linear)
//                 - 2'b01: RELU (max(0, x))
//                 - 2'b10: GELU (Piecewise Linear Transformer Approximation)
//                 - 2'b11: CLAMP / CLIP
//              3. INT32 to INT8 Quantization with rounding and symmetric saturation [-128, 127].
// =============================================================================

`timescale 1ns / 1ps

module vector_unit #(
    parameter ARRAY_SIZE = 8
)(
    input  wire                               clk,
    input  wire                               rst_n,
    input  wire                               enable,

    // Configuration
    input  wire [1:0]                         act_func,       // 00=Pass, 01=ReLU, 10=GELU, 11=Clip
    input  wire                               bias_en,        // Enable bias addition
    input  wire [3:0]                         quant_shift,    // Right shift for INT8 quantization (0..15)

    // Data Inputs: Parallel 32-bit accumulated sums from deskew unit
    input  wire [ARRAY_SIZE*32-1:0]           sum_in_flat,

    // Bias Inputs: Parallel 32-bit bias vector
    input  wire [ARRAY_SIZE*32-1:0]           bias_in_flat,

    // Data Outputs: Parallel 32-bit unquantized vector
    output reg  [ARRAY_SIZE*32-1:0]           data_out_32_flat,

    // Data Outputs: Parallel 8-bit quantized vector (for direct next-layer feedback)
    output reg  [ARRAY_SIZE*8-1:0]            data_out_8_flat,

    output reg                                valid_out
);

    // Act function constants
    localparam ACT_PASS = 2'b00;
    localparam ACT_RELU = 2'b01;
    localparam ACT_GELU = 2'b10;
    localparam ACT_CLIP = 2'b11;

    // Combinational processing per channel
    wire signed [31:0] ch_in      [ARRAY_SIZE-1:0];
    wire signed [31:0] ch_bias    [ARRAY_SIZE-1:0];
    wire signed [31:0] ch_biased  [ARRAY_SIZE-1:0];
    reg  signed [31:0] ch_act     [ARRAY_SIZE-1:0];
    wire signed [31:0] ch_rounded [ARRAY_SIZE-1:0];
    reg  signed [7:0]  ch_quant   [ARRAY_SIZE-1:0];

    genvar i;
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : gen_vpu_channel
            assign ch_in[i]   = sum_in_flat[i*32 +: 32];
            assign ch_bias[i] = bias_in_flat[i*32 +: 32];

            // 1. Bias Addition
            assign ch_biased[i] = bias_en ? (ch_in[i] + ch_bias[i]) : ch_in[i];

            // 2. Activation Function Execution
            always @(*) begin
                case (act_func)
                    ACT_RELU: begin
                        ch_act[i] = (ch_biased[i] < 32'sd0) ? 32'sd0 : ch_biased[i];
                    end

                    ACT_GELU: begin
                        // Piecewise Linear (PWL) GELU approximation:
                        // Deep negative: x < -256 => 0
                        // Moderate negative: -256 <= x < -64 => slight dip ~ (x >>> 4)
                        // Shallow negative: -64 <= x < 0 => dip ~ (x >>> 3)
                        // Linear positive: 0 <= x < 256 => smooth slope (3*x >>> 2)
                        // Saturated positive: x >= 256 => x
                        if (ch_biased[i] < -32'sd256) begin
                            ch_act[i] = 32'sd0;
                        end else if (ch_biased[i] < -32'sd64) begin
                            ch_act[i] = ch_biased[i] >>> 4;
                        end else if (ch_biased[i] < 32'sd0) begin
                            ch_act[i] = ch_biased[i] >>> 3;
                        end else if (ch_biased[i] < 32'sd256) begin
                            ch_act[i] = (ch_biased[i] * 32'sd3) >>> 2;
                        end else begin
                            ch_act[i] = ch_biased[i];
                        end
                    end

                    ACT_CLIP: begin
                        if (ch_biased[i] > 32'sd127)
                            ch_act[i] = 32'sd127;
                        else if (ch_biased[i] < -32'sd128)
                            ch_act[i] = -32'sd128;
                        else
                            ch_act[i] = ch_biased[i];
                    end

                    default: begin // ACT_PASS
                        ch_act[i] = ch_biased[i];
                    end
                endcase
            end

            // 3. INT32 to INT8 Quantization with Half-Up Rounding & Saturation
            // Round offset = (1 << (quant_shift - 1)) when quant_shift > 0
            wire signed [31:0] round_offset = (quant_shift > 4'd0) ? (32'sd1 <<< (quant_shift - 1)) : 32'sd0;
            assign ch_rounded[i] = (ch_act[i] + round_offset) >>> quant_shift;

            always @(*) begin
                if (ch_rounded[i] > 32'sd127)
                    ch_quant[i] = 8'sd127;
                else if (ch_rounded[i] < -32'sd128)
                    ch_quant[i] = -8'sd128;
                else
                    ch_quant[i] = ch_rounded[i][7:0];
            end
        end
    endgenerate

    // Registered pipeline stage for VPU output
    integer c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out_32_flat <= {(ARRAY_SIZE*32){1'b0}};
            data_out_8_flat  <= {(ARRAY_SIZE*8){1'b0}};
            valid_out        <= 1'b0;
        end else if (enable) begin
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                data_out_32_flat[c*32 +: 32] <= ch_act[c];
                data_out_8_flat[c*8 +: 8]   <= ch_quant[c];
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
