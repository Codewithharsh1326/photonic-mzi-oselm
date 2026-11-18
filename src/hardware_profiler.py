"""
Electro-Optic Hardware Resource, Latency & Energy Profiler
==========================================================
Models hardware partitioning between a passive Silicon Photonic Clements MZI Mesh
and a Xilinx Zynq UltraScale+ ZCU104 FPGA (xczu7ev-ffvc1156-2-e) operating at
f_clk = 250 MHz with a 24-bit fixed-point (Q8.16) datapath.

Target FPGA Device Specifications (Zynq UltraScale+ XCZU7EV):
- Programmable Logic LUTs: 230,400
- Flip-Flops (FFs):        460,800
- DSP48E2 Slices:          1,728
- Block RAM (36Kb blocks): 312 (11.23 Mb total)
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ZCU104Specs:
    """Available programmable logic resources on Xilinx Zynq UltraScale+ ZCU104."""
    device_name: str = "Xilinx Zynq UltraScale+ XCZU7EV (ZCU104)"
    total_luts: int = 230_400
    total_dsp48e2: int = 1_728
    total_bram36k: int = 312
    clock_mhz: float = 250.0


class ElectroOpticHardwareProfiler:
    """
    Analytical hardware resource, latency, and dynamic energy model comparing:
      1. Pure Digital FPGA OS-ELM (Hidden random projection + Readout on FPGA)
      2. Hybrid Photonic-Digital OS-ELM (Hidden projection on passive SOI MZI mesh;
         Readout + Sherman-Morrison updates on FPGA)
    """

    def __init__(self, fpga: ZCU104Specs | None = None):
        self.fpga = fpga or ZCU104Specs()

    def profile_architecture(
        self,
        n_inputs: int,
        n_hidden: int,
        n_classes: int,
        optical_delay_ps: float,
        unroll_factor: int = 32,
    ) -> Dict[str, Any]:
        """
        Compute comparative MAC counts, memory footprints, FPGA resource utilization,
        single-sample inference latency, and dynamic energy per sample.
        """
        d, L, C = n_inputs, n_hidden, n_classes
        clk_period_ns = 1000.0 / self.fpga.clock_mhz  # 4.0 ns at 250 MHz

        # --- 1. Multiply-Accumulate (MAC) Operations ---
        hidden_macs = d * L                  # Replaced by passive optical MZI mesh in Hybrid
        readout_infer_macs = L * C           # Digital readout dot products
        total_digital_infer_macs = hidden_macs + readout_infer_macs
        hybrid_digital_infer_macs = readout_infer_macs
        infer_mac_reduction_pct = 100.0 * (1.0 - hybrid_digital_infer_macs / total_digital_infer_macs)

        # Sherman-Morrison sequential training update MACs in digital domain:
        # g = P*h (L^2), h^T*g (L), P - k*g^T (L^2), e = y - beta^T*h (L*C), beta + k*e^T (L*C)
        sm_update_macs = 2 * (L ** 2) + L + 2 * (L * C)

        # --- 2. On-Chip Memory Footprint (24-bit = 3 bytes per Q8.16 word) ---
        bytes_per_word = 3
        hidden_weights_kb = (d * L + L) * bytes_per_word / 1024.0
        readout_weights_kb = (L * C) * bytes_per_word / 1024.0
        covariance_p_kb = (L * L) * bytes_per_word / 1024.0

        pure_digital_mem_kb = hidden_weights_kb + readout_weights_kb + covariance_p_kb
        hybrid_digital_mem_kb = readout_weights_kb + covariance_p_kb

        # --- 3. FPGA Resource Utilization (DSP48E2, BRAM36K, CLB LUTs) ---
        # Pure Digital requires parallel DSP pipelines for d*L hidden MACs + sigmoid LUTs + readout
        digital_dsp = min(self.fpga.total_dsp48e2, unroll_factor * 4 + (L // 4))
        hybrid_dsp = min(self.fpga.total_dsp48e2, unroll_factor + (L // 16))

        digital_bram = int(np.ceil(pure_digital_mem_kb / 4.0)) + 4
        hybrid_bram = int(np.ceil(hybrid_digital_mem_kb / 4.0)) + 1

        digital_luts = 4_200 + int(18.5 * L) + int(1.2 * d * (L // 16))
        hybrid_luts = 2_450 + int(11.0 * L)

        # --- 4. Single-Sample Inference Latency (us) ---
        # Pure Digital: AXI DMA ingress + d*L hidden MAC cycles + LUT activation + L*C readout
        digital_hidden_cycles = int(np.ceil((d * L) / unroll_factor)) + 14
        digital_readout_cycles = int(np.ceil((L * C) / unroll_factor)) + 10
        digital_total_cycles = digital_hidden_cycles + digital_readout_cycles
        digital_infer_latency_us = (digital_total_cycles * clk_period_ns) / 1000.0

        # Hybrid Electro-Optic:
        # Optical propagation (~25 ps) + high-speed Ge PD/TIA/ADC pipeline (~18 ns) + FPGA readout
        e_o_conversion_ns = 18.0
        hybrid_readout_cycles = int(np.ceil((L * C) / unroll_factor)) + 8
        hybrid_infer_latency_us = (
            (optical_delay_ps * 1e-3) + e_o_conversion_ns + (hybrid_readout_cycles * clk_period_ns)
        ) / 1000.0

        # Sequential Sherman-Morrison Single-Sample Update Latency (us)
        sm_cycles = int(np.ceil(sm_update_macs / unroll_factor)) + 28
        sm_update_latency_us = (sm_cycles * clk_period_ns) / 1000.0

        # --- 5. Dynamic Energy per Sample Inference (pJ) ---
        # 16nm FinFET Zynq UltraScale+ Q8.16 MAC + BRAM fetch: ~14.5 pJ/MAC
        # Passive SOI MZI mesh optical matrix multiplication: 0 pJ static mesh tuning +
        # ~6.5 pJ/channel laser/modulator/TIA/ADC optoelectronic front-end overhead
        pj_per_fpga_mac = 14.5
        pj_per_sram_read = 8.2
        pj_per_eo_channel = 6.5

        digital_infer_energy_pj = (
            total_digital_infer_macs * (pj_per_fpga_mac + pj_per_sram_read)
        )
        hybrid_infer_energy_pj = (
            (d + L) * pj_per_eo_channel
            + hybrid_digital_infer_macs * (pj_per_fpga_mac + pj_per_sram_read)
        )
        energy_reduction_pct = 100.0 * (1.0 - hybrid_infer_energy_pj / digital_infer_energy_pj)

        return {
            "n_inputs": d,
            "n_hidden": L,
            "n_classes": C,
            "hidden_macs_offloaded": hidden_macs,
            "readout_infer_macs": readout_infer_macs,
            "total_digital_infer_macs": total_digital_infer_macs,
            "hybrid_digital_infer_macs": hybrid_digital_infer_macs,
            "infer_mac_reduction_pct": round(infer_mac_reduction_pct, 2),
            "sm_update_macs": sm_update_macs,
            "pure_digital_mem_kb": round(pure_digital_mem_kb, 2),
            "hybrid_digital_mem_kb": round(hybrid_digital_mem_kb, 2),
            "hidden_weights_saved_kb": round(hidden_weights_kb, 2),
            "digital_dsp48e2": digital_dsp,
            "hybrid_dsp48e2": hybrid_dsp,
            "dsp_reduction_pct": round(100.0 * (1.0 - hybrid_dsp / digital_dsp), 2),
            "digital_bram36k": digital_bram,
            "hybrid_bram36k": hybrid_bram,
            "digital_luts": digital_luts,
            "hybrid_luts": hybrid_luts,
            "hybrid_lut_utilization_pct": round(100.0 * hybrid_luts / self.fpga.total_luts, 2),
            "optical_propagation_ps": round(optical_delay_ps, 2),
            "digital_infer_latency_us": round(digital_infer_latency_us, 3),
            "hybrid_infer_latency_us": round(hybrid_infer_latency_us, 3),
            "latency_speedup_x": round(digital_infer_latency_us / hybrid_infer_latency_us, 2),
            "sm_update_latency_us": round(sm_update_latency_us, 2),
            "digital_infer_energy_nj": round(digital_infer_energy_pj / 1000.0, 2),
            "hybrid_infer_energy_nj": round(hybrid_infer_energy_pj / 1000.0, 2),
            "energy_reduction_pct": round(energy_reduction_pct, 2),
        }


import numpy as np
