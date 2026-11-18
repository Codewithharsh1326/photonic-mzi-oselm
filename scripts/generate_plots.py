"""
Publication-Grade Figure Generator for Hybrid Photonic-Digital OS-ELM
=====================================================================
Generates 300-DPI diagnostic and benchmark plots in `results/`:
1. fig1_mzi_unitary_and_transmission.png
2. fig2_accuracy_vs_modes_fp64_q8_16.png
3. fig3_thermo_optic_drift_adaptation.png
4. fig4_fpga_dsp_bram_energy_savings.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
})

NAVY = "#0F2C59"
TEAL = "#008080"
CORAL = "#D9534F"
AMBER = "#E67E22"
SLATE = "#5D6D7E"


def plot_all_figures(
    results_dir: Path,
    U_ideal: np.ndarray,
    H_sample: np.ndarray,
    df_modes: pd.DataFrame,
    df_drift: pd.DataFrame,
    df_hw: pd.DataFrame,
) -> None:
    """Render all four benchmark figures into `results_dir`."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 1: Passive 128-Mode Clements MZI Unitary Matrix & PD Activations
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2), constrained_layout=True)

    # Panel A: Power transmission matrix |U_{i,j}|^2 (top-left 64x64 subblock)
    power_mat = np.abs(U_ideal[:64, :64]) ** 2
    im0 = axes[0].imshow(power_mat, cmap="viridis", aspect="auto", interpolation="nearest")
    axes[0].set_title(r"(a) MZI Transmission $|U_{m,n}|^2$ (64x64)")
    axes[0].set_xlabel("Input Waveguide Mode Index $n$")
    axes[0].set_ylabel("Output Waveguide Mode Index $m$")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label("Optical Transmission")

    # Panel B: Optical phase distribution arg(U_{i,j}) across all 128x128 modes
    phases = np.angle(U_ideal).flatten()
    axes[1].hist(phases, bins=45, density=True, color=TEAL, alpha=0.82, edgecolor="white")
    axes[1].axhline(1.0 / (2.0 * np.pi), color=CORAL, linestyle="--", linewidth=1.8, label=r"Uniform $1/(2\pi)$")
    axes[1].set_title(r"(b) Untuned Phase Distribution $\angle U_{m,n}$")
    axes[1].set_xlabel("Optical Phase Shift (radians)")
    axes[1].set_ylabel("Probability Density")
    axes[1].set_xlim(-np.pi, np.pi)
    axes[1].legend(loc="upper right", frameon=True)

    # Panel C: Ge-on-Si Photodiode Square-Law Activation Distribution
    h_vals = H_sample.flatten()
    axes[2].hist(h_vals, bins=45, density=True, color=NAVY, alpha=0.82, edgecolor="white", label=r"PD Output $h_j$")
    axes[2].set_title(r"(c) Ge Photodiode Nonlinearity $I = R_{\mathrm{PD}}|E_{\mathrm{out}}|^2$")
    axes[2].set_xlabel("Normalized Photocurrent Activation (Q8.16)")
    axes[2].set_ylabel("Probability Density")
    axes[2].legend(loc="upper right", frameon=True)

    fig.savefig(results_dir / "fig1_mzi_unitary_and_transmission.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Classification Accuracy vs. MZI Waveguide Modes (FP64 vs Q8.16)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)

    modes = df_modes["n_modes"].values
    ax.plot(
        modes,
        df_modes["photonic_fp64_acc_pct"].values,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color=NAVY,
        label="Hybrid Photonic MZI + Digital OS-ELM (IEEE-754 FP64)",
    )
    ax.plot(
        modes,
        df_modes["photonic_q8_16_acc_pct"].values,
        marker="s",
        linewidth=2.2,
        markersize=7,
        linestyle="--",
        color=TEAL,
        label="Hybrid Photonic MZI + Zynq FPGA OS-ELM (24-bit Q8.16)",
    )
    ax.plot(
        modes,
        df_modes["digital_q8_16_acc_pct"].values,
        marker="^",
        linewidth=1.8,
        markersize=6.5,
        linestyle=":",
        color=AMBER,
        label="Pure Digital FPGA OS-ELM Baseline (24-bit Q8.16)",
    )

    row_128 = df_modes[df_modes["n_modes"] == 128].iloc[0]
    ax.scatter([128], [row_128["photonic_q8_16_acc_pct"]], s=110, color=CORAL, zorder=6)
    ax.annotate(
        f"M=128 Design Point\n{row_128['photonic_q8_16_acc_pct']:.2f}% (Q8.16)",
        xy=(128, row_128["photonic_q8_16_acc_pct"]),
        xytext=(142, row_128["photonic_q8_16_acc_pct"] - 3.5),
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.4),
        fontsize=8.5,
        fontweight="bold",
        color=CORAL,
    )

    ax.set_title("Classification Accuracy vs. Photonic MZI Mesh Mode Count ($M$)")
    ax.set_xlabel("Number of Waveguide Modes / Hidden Neurons ($M = L$)")
    ax.set_ylabel("Test Classification Accuracy (%)")
    ax.set_xticks(modes)
    ax.set_ylim(76.0, 101.0)
    ax.legend(loc="lower right", frameon=True)

    fig.savefig(results_dir / "fig2_accuracy_vs_modes_fp64_q8_16.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Real-Time Digital Sherman-Morrison Tracking Under Thermal Drift
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)

    step = df_drift["stream_step"].values
    temp_k = df_drift["delta_T_kelvin"].values
    acc_adaptive = df_drift["adaptive_q8_16_acc_pct"].values
    acc_static = df_drift["static_readout_acc_pct"].values

    ax1.plot(
        step,
        acc_adaptive,
        color=NAVY,
        linewidth=2.3,
        marker="o",
        markersize=5,
        label=r"Adaptive FPGA Sherman-Morrison Readout ($\lambda_{\mathrm{RLS}}=0.985$, Q8.16)",
    )
    ax1.plot(
        step,
        acc_static,
        color=CORAL,
        linewidth=2.1,
        linestyle="--",
        marker="x",
        markersize=5,
        label="Static Unadapted Readout (Batch ELM without Online Tracking)",
    )
    ax1.set_title("Digital Sherman-Morrison Compensation of Silicon Thermo-Optic Phase Drift")
    ax1.set_xlabel("Streaming Optical Frame Index ($k$)")
    ax1.set_ylabel("Classification Accuracy (%)")
    ax1.set_ylim(25.0, 103.0)

    ax2 = ax1.twinx()
    ax2.fill_between(step, 0, temp_k, color=AMBER, alpha=0.16, label=r"Chip Temp. Drift $\Delta T(t)$")
    ax2.plot(step, temp_k, color=AMBER, linestyle=":", linewidth=1.6)
    ax2.set_ylabel(r"SOI Waveguide Temperature Drift $\Delta T$ (K)", color=AMBER)
    ax2.tick_params(axis="y", labelcolor=AMBER)
    ax2.set_ylim(0.0, max(temp_k) * 1.25)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True)

    fig.savefig(results_dir / "fig3_thermo_optic_drift_adaptation.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 4: Zynq UltraScale+ ZCU104 Hardware Resource & Latency Comparison
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    x_idx = np.arange(len(df_hw))
    width = 0.36
    labels = [f"L={m}" for m in df_hw["n_hidden"].values]

    axes[0].bar(x_idx - width / 2, df_hw["digital_infer_latency_us"], width, color=SLATE, label="Pure Digital FPGA")
    axes[0].bar(x_idx + width / 2, df_hw["hybrid_infer_latency_us"], width, color=TEAL, label="Hybrid Photonic-FPGA")
    axes[0].set_title("(a) Single-Sample Inference Latency")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel(r"Inference Latency ($\mu$s)")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].bar(x_idx - width / 2, df_hw["digital_dsp48e2"], width, color=SLATE, label="Pure Digital FPGA")
    axes[1].bar(x_idx + width / 2, df_hw["hybrid_dsp48e2"], width, color=NAVY, label="Hybrid Photonic-FPGA")
    axes[1].set_title("(b) ZCU104 DSP48E2 Multiplier Slices")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Active DSP48E2 Slices")
    axes[1].legend(loc="upper left", frameon=True)

    axes[2].bar(x_idx - width / 2, df_hw["digital_infer_energy_nj"], width, color=CORAL, label="Pure Digital FPGA")
    axes[2].bar(x_idx + width / 2, df_hw["hybrid_infer_energy_nj"], width, color=TEAL, label="Hybrid Photonic-FPGA")
    axes[2].set_title("(c) Dynamic Inference Energy / Sample")
    axes[2].set_xticks(x_idx)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Energy per Sample (nJ)")
    axes[2].legend(loc="upper left", frameon=True)

    fig.savefig(results_dir / "fig4_fpga_dsp_bram_energy_savings.png")
    plt.close(fig)
