// =============================================================================
// File: dma_axi_master.v
// Module: dma_axi_master
// Description: AMBA AXI4 Master DMA Controller for Apex-TPU.
//              Handles high-throughput burst reads and writes between external
//              DRAM / System Memory and on-chip Unified Buffer / Weight Buffer.
//              Supports 64-bit data bus bursts up to 256 beats.
// =============================================================================

`timescale 1ns / 1ps

module dma_axi_master #(
    parameter AXI_ADDR_WIDTH = 32,
    parameter AXI_DATA_WIDTH = 64,
    parameter AXI_ID_WIDTH   = 4
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // Command Interface from TPU Controller
    input  wire                      cmd_start,
    input  wire                      cmd_dir,       // 0 = Read (DRAM -> TPU), 1 = Write (TPU -> DRAM)
    input  wire                      cmd_target,    // 0 = Unified Buffer, 1 = Weight Buffer
    input  wire [AXI_ADDR_WIDTH-1:0] cmd_ext_addr,  // System memory base byte address
    input  wire [7:0]                cmd_int_addr,  // Internal SRAM start row index
    input  wire [7:0]                cmd_length,    // Number of 64-bit rows to transfer
    output reg                       cmd_done,
    output reg                       cmd_busy,

    // Internal SRAM Buffer Ports
    // Unified Buffer Port A Interface
    output reg                       ub_en,
    output reg                       ub_we,
    output reg  [7:0]                ub_addr,
    output reg  [AXI_DATA_WIDTH-1:0] ub_wdata,
    input  wire [AXI_DATA_WIDTH-1:0] ub_rdata,

    // Weight Buffer Write Interface
    output reg                       wb_we,
    output reg  [5:0]                wb_addr,
    output reg  [AXI_DATA_WIDTH-1:0] wb_wdata,

    // -------------------------------------------------------------------------
    // AXI4 Master Interface
    // -------------------------------------------------------------------------
    // Read Address Channel
    output reg  [AXI_ID_WIDTH-1:0]   m_axi_arid,
    output reg  [AXI_ADDR_WIDTH-1:0] m_axi_araddr,
    output reg  [7:0]                m_axi_arlen,
    output reg  [2:0]                m_axi_arsize,
    output reg  [1:0]                m_axi_arburst,
    output reg                       m_axi_arvalid,
    input  wire                      m_axi_arready,

    // Read Data Channel
    input  wire [AXI_ID_WIDTH-1:0]   m_axi_rid,
    input  wire [AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input  wire [1:0]                m_axi_rresp,
    input  wire                      m_axi_rlast,
    input  wire                      m_axi_rvalid,
    output reg                       m_axi_rready,

    // Write Address Channel
    output reg  [AXI_ID_WIDTH-1:0]   m_axi_awid,
    output reg  [AXI_ADDR_WIDTH-1:0] m_axi_awaddr,
    output reg  [7:0]                m_axi_awlen,
    output reg  [2:0]                m_axi_awsize,
    output reg  [1:0]                m_axi_awburst,
    output reg                       m_axi_awvalid,
    input  wire                      m_axi_awready,

    // Write Data Channel
    output wire [AXI_DATA_WIDTH-1:0]     m_axi_wdata,
    output reg  [(AXI_DATA_WIDTH/8)-1:0] m_axi_wstrb,
    output reg                           m_axi_wlast,
    output reg                           m_axi_wvalid,
    input  wire                          m_axi_wready,

    // Write Response Channel
    input  wire [AXI_ID_WIDTH-1:0]   m_axi_bid,
    input  wire [1:0]                m_axi_bresp,
    input  wire                      m_axi_bvalid,
    output reg                       m_axi_bready
);

    // FSM States
    localparam STATE_IDLE       = 3'd0;
    localparam STATE_AR         = 3'd1;
    localparam STATE_RDATA      = 3'd2;
    localparam STATE_AW         = 3'd3;
    localparam STATE_WDATA_READ = 3'd4;
    localparam STATE_WDATA      = 3'd5;
    localparam STATE_BRESP      = 3'd6;

    reg [2:0] state;
    reg [7:0] beat_cnt;
    reg [7:0] current_int_addr;
    reg       reg_target;

    assign m_axi_wdata = ub_rdata;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= STATE_IDLE;
            cmd_done         <= 1'b0;
            cmd_busy         <= 1'b0;
            beat_cnt         <= 8'd0;
            current_int_addr <= 8'd0;
            reg_target       <= 1'b0;

            ub_en            <= 1'b0;
            ub_we            <= 1'b0;
            ub_addr          <= 8'd0;
            ub_wdata         <= {AXI_DATA_WIDTH{1'b0}};

            wb_we            <= 1'b0;
            wb_addr          <= 6'd0;
            wb_wdata         <= {AXI_DATA_WIDTH{1'b0}};

            m_axi_arid       <= {AXI_ID_WIDTH{1'b0}};
            m_axi_araddr     <= {AXI_ADDR_WIDTH{1'b0}};
            m_axi_arlen      <= 8'd0;
            m_axi_arsize     <= 3'b011; // 8 bytes (64 bits)
            m_axi_arburst    <= 2'b01;  // INCR burst
            m_axi_arvalid    <= 1'b0;
            m_axi_rready     <= 1'b0;

            m_axi_awid       <= {AXI_ID_WIDTH{1'b0}};
            m_axi_awaddr     <= {AXI_ADDR_WIDTH{1'b0}};
            m_axi_awlen      <= 8'd0;
            m_axi_awsize     <= 3'b011; // 8 bytes (64 bits)
            m_axi_awburst    <= 2'b01;  // INCR burst
            m_axi_awvalid    <= 1'b0;
            m_axi_wstrb      <= {(AXI_DATA_WIDTH/8){1'b1}};
            m_axi_wlast      <= 1'b0;
            m_axi_wvalid     <= 1'b0;
            m_axi_bready     <= 1'b0;
        end else begin
            // Defaults for single-cycle strobes
            cmd_done <= 1'b0;
            ub_en    <= 1'b0;
            ub_we    <= 1'b0;
            wb_we    <= 1'b0;

            case (state)
                STATE_IDLE: begin
                    cmd_busy <= 1'b0;
                    if (cmd_start) begin
                        cmd_busy         <= 1'b1;
                        reg_target       <= cmd_target;
                        current_int_addr <= cmd_int_addr;
                        beat_cnt         <= cmd_length;

                        if (!cmd_dir) begin
                            // Read from DRAM (AR phase)
                            m_axi_araddr  <= cmd_ext_addr;
                            m_axi_arlen   <= cmd_length - 8'd1;
                            m_axi_arvalid <= 1'b1;
                            state         <= STATE_AR;
                        end else begin
                            // Write to DRAM (AW phase)
                            m_axi_awaddr     <= cmd_ext_addr;
                            m_axi_awlen      <= cmd_length - 8'd1;
                            m_axi_awvalid    <= 1'b1;
                            // Kick off first SRAM read for write data
                            ub_en            <= 1'b1;
                            ub_we            <= 1'b0;
                            ub_addr          <= cmd_int_addr;
                            current_int_addr <= cmd_int_addr + 8'd1;
                            state            <= STATE_AW;
                        end
                    end
                end

                // -------------------------------------------------------------
                // Read Channel (DRAM -> On-Chip SRAM)
                // -------------------------------------------------------------
                STATE_AR: begin
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        state         <= STATE_RDATA;
                    end
                end

                STATE_RDATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        // Latch read beat into SRAM
                        if (!reg_target) begin
                            // Write to Unified Buffer
                            ub_en    <= 1'b1;
                            ub_we    <= 1'b1;
                            ub_addr  <= current_int_addr;
                            ub_wdata <= m_axi_rdata;
                        end else begin
                            // Write to Weight Buffer
                            wb_we    <= 1'b1;
                            wb_addr  <= current_int_addr[5:0];
                            wb_wdata <= m_axi_rdata;
                        end

                        current_int_addr <= current_int_addr + 8'd1;

                        if (m_axi_rlast) begin
                            m_axi_rready <= 1'b0;
                            cmd_done     <= 1'b1;
                            cmd_busy     <= 1'b0;
                            state        <= STATE_IDLE;
                        end
                    end
                end

                // -------------------------------------------------------------
                // Write Channel (Unified Buffer -> DRAM)
                // -------------------------------------------------------------
                STATE_AW: begin
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b1;
                        m_axi_wlast   <= (beat_cnt == 8'd1);
                        if (beat_cnt > 8'd1) begin
                            ub_en            <= 1'b1;
                            ub_we            <= 1'b0;
                            ub_addr          <= current_int_addr;
                            current_int_addr <= current_int_addr + 8'd1;
                        end
                        state <= STATE_WDATA;
                    end
                end

                STATE_WDATA: begin
                    m_axi_wvalid <= 1'b1;
                    m_axi_wlast  <= (beat_cnt == 8'd1);

                    if (m_axi_wready && m_axi_wvalid) begin
                        beat_cnt <= beat_cnt - 8'd1;

                        if (beat_cnt > 8'd2) begin
                            ub_en            <= 1'b1;
                            ub_we            <= 1'b0;
                            ub_addr          <= current_int_addr;
                            current_int_addr <= current_int_addr + 8'd1;
                        end else if (beat_cnt == 8'd1) begin
                            m_axi_wvalid <= 1'b0;
                            m_axi_wlast  <= 1'b0;
                            m_axi_bready <= 1'b1;
                            state        <= STATE_BRESP;
                        end
                    end
                end

                STATE_BRESP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 1'b0;
                        cmd_done     <= 1'b1;
                        cmd_busy     <= 1'b0;
                        state        <= STATE_IDLE;
                    end
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
