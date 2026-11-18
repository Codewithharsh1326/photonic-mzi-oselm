#!/usr/bin/env python3
"""
Hybrid Photonic-Digital OS-ELM Hardware-Software Co-Design Simulator
====================================================================
Main execution pipeline that:
1. Simulates an untuned, passive Silicon-on-Insulator (SOI) Clements Mach-Zehnder
   Interferometer (MZI) mesh (lambda = 1550 nm) as a zero-power optical random
   projection hidden layer with Germanium photodiode square-law nonlinearity.
2. Executes sequential rank-1 Sherman-Morrison recursive least squares (RLS) updates
   in a 24-bit signed fixed-point (Q8.16 / ap_fixed<24,8>) digital backend modeling
   a Xilinx Zynq UltraScale+ (ZCU104) FPGA.
3. Evaluates classification accuracy across waveguide mode counts M in {32..256},
   quantifies real-time digital adaptation under silicon thermo-optic phase drift
   (dn_eff/dT = 1.86e-4 K^-1), and profiles FPGA DSP48E2/BRAM/latency/energy savings.

Usage:
    python3 main.py --dataset optical_digits --modes 128
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.mzi_mesh import PhotonicMeshConfig, ClementsMZIMesh
from src.photodetector import PhotodetectorConfig, PhotodetectorArray
from src.digital_oselm import ShermanMorrisonReadout, DigitalRandomHiddenLayer
from src.hardware_profiler import ElectroOpticHardwareProfiler
from src.dataset_loader import load_benchmark_dataset
from scripts.generate_plots import plot_all_figures


def run_hidden_dimension_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mode_list: list[int],
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate FP64 and 24-bit Q8.16 Photonic OS-ELM vs. Pure Digital OS-ELM across mode counts."""
    records = []
    n_inputs = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    for M in mode_list:
        mesh_cfg = PhotonicMeshConfig(n_modes=M, seed=seed)
        mesh = ClementsMZIMesh(mesh_cfg)
        pd_array = PhotodetectorArray(PhotodetectorConfig(snr_db=34.0, adc_bits=12, seed=seed + 59))

        # Propagate through passive MZI mesh + square-law photodetection
        H_train_opt = pd_array.detect(mesh.propagate(X_train))
        H_test_opt = pd_array.detect(mesh.propagate(X_test))

        # 1. Hybrid Photonic + FP64 Sherman-Morrison Readout
        model_fp64 = ShermanMorrisonReadout(
            n_hidden=M, n_classes=n_classes, reg_lambda=1.5, use_fixed_point=False
        )
        t0 = time.perf_counter()
        model_fp64.partial_fit(H_train_opt, y_train)
        fp64_train_ms = (time.perf_counter() - t0) * 1000.0
        acc_fp64 = model_fp64.score(H_test_opt, y_test) * 100.0

        # 2. Hybrid Photonic + 24-Bit Fixed-Point (Q8.16) FPGA Sherman-Morrison Readout
        model_q8_16 = ShermanMorrisonReadout(
            n_hidden=M, n_classes=n_classes, reg_lambda=1.5, use_fixed_point=True
        )
        model_q8_16.partial_fit(H_train_opt, y_train)
        acc_q8_16 = model_q8_16.score(H_test_opt, y_test) * 100.0

        # Quantization fidelity between FP64 and Q8.16 readout weights
        weight_rmse = float(np.sqrt(np.mean((model_fp64.beta - model_q8_16.beta) ** 2)))

        # 3. Pure Digital OS-ELM Baseline (Q8.16)
        dig_hidden = DigitalRandomHiddenLayer(n_inputs=n_inputs, n_hidden=M, seed=seed, use_fixed_point=True)
        H_train_dig = dig_hidden.project(X_train)
        H_test_dig = dig_hidden.project(X_test)
        model_dig = ShermanMorrisonReadout(
            n_hidden=M, n_classes=n_classes, reg_lambda=1.5, use_fixed_point=True
        )
        model_dig.partial_fit(H_train_dig, y_train)
        acc_dig = model_dig.score(H_test_dig, y_test) * 100.0

        records.append({
            "n_modes": M,
            "n_mzis": mesh.n_mzis,
            "chip_length_mm": round(mesh.chip_length_mm, 2),
            "optical_delay_ps": round(mesh.optical_propagation_delay_ps, 2),
            "insertion_loss_db": round(mesh.total_insertion_loss_db, 2),
            "unitarity_error_fro": float(f"{mesh.verify_unitarity_error():.2e}"),
            "photonic_fp64_acc_pct": round(acc_fp64, 2),
            "photonic_q8_16_acc_pct": round(acc_q8_16, 2),
            "digital_q8_16_acc_pct": round(acc_dig, 2),
            "q8_16_degradation_pct": round(acc_fp64 - acc_q8_16, 2),
            "beta_q8_16_rmse": round(weight_rmse, 5),
            "host_sim_time_ms": round(fp64_train_ms, 1),
        })

    return pd.DataFrame(records)


def run_thermo_optic_drift_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_modes: int = 128,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate continuous silicon waveguide temperature drift Delta_T from 0.0 K to +2.25 K
    over a 16-epoch streaming sequence, comparing:
      - Static Readout (trained once at nominal chip temperature Delta_T = 0 K)
      - Online Sequential Sherman-Morrison Readout with forgetting factor lambda_RLS = 0.985
        running in 24-bit Q8.16 fixed-point on the Zynq UltraScale+ FPGA backend.
    """
    n_classes = len(np.unique(y_train))
    mesh = ClementsMZIMesh(PhotonicMeshConfig(n_modes=n_modes, phase_arm_length_um=90.0, seed=seed))
    pd_array = PhotodetectorArray(PhotodetectorConfig(snr_db=34.0, adc_bits=12, seed=seed + 99))

    # Initial calibration at nominal chip temperature Delta_T = 0.0 K
    H_init = pd_array.detect(mesh.propagate(X_train[:1200], delta_T_kelvin=0.0))
    y_init = y_train[:1200]

    static_model = ShermanMorrisonReadout(
        n_hidden=n_modes, n_classes=n_classes, reg_lambda=1.5, forgetting_factor=1.0, use_fixed_point=True
    )
    static_model.partial_fit(H_init, y_init)

    adaptive_model = ShermanMorrisonReadout(
        n_hidden=n_modes, n_classes=n_classes, reg_lambda=1.5, forgetting_factor=1.0, use_fixed_point=True
    )
    adaptive_model.partial_fit(H_init, y_init)
    # Enable RLS exponential forgetting for streaming thermal drift tracking
    adaptive_model.forgetting_factor = 0.985

    n_epochs = 16
    max_delta_T = 2.25  # Kelvin

    drift_records = []
    for ep in range(n_epochs):
        delta_T = max_delta_T * (ep / (n_epochs - 1))

        H_adapt = pd_array.detect(mesh.propagate(X_train[:1200], delta_T_kelvin=delta_T))
        y_adapt = y_train[:1200]
        H_eval = pd_array.detect(mesh.propagate(X_test, delta_T_kelvin=delta_T))
        y_eval = y_test

        if ep > 0:
            adaptive_model.partial_fit(H_adapt, y_adapt)

        static_acc = static_model.score(H_eval, y_eval) * 100.0
        adaptive_acc = adaptive_model.score(H_eval, y_eval) * 100.0

        drift_records.append({
            "stream_step": (ep + 1) * 150,
            "delta_T_kelvin": round(delta_T, 2),
            "static_readout_acc_pct": round(static_acc, 2),
            "adaptive_q8_16_acc_pct": round(adaptive_acc, 2),
            "accuracy_gain_pct": round(adaptive_acc - static_acc, 2),
        })

    return pd.DataFrame(drift_records)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Photonic-Digital OS-ELM Co-Design Simulator (Passive MZI Mesh + Zynq FPGA)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="optical_digits",
        choices=["optical_digits", "sklearn_digits"],
        help="Benchmark dataset to evaluate (default: optical_digits)",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=128,
        help="Target optical waveguide mode count M for detailed profiling (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility (default: 42)",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" HYBRID PHOTONIC-DIGITAL OS-ELM HARDWARE-SOFTWARE CO-DESIGN SIMULATOR")
    print(" Optical Layer : Passive SOI Clements MZI Mesh (lambda = 1550 nm, 0 mW Tuning)")
    print(" Digital Layer : Xilinx Zynq UltraScale+ ZCU104 (24-Bit Q8.16 Sherman-Morrison)")
    print("=" * 78)

    # 1. Load 64-Port Optical Benchmark Dataset
    X_train, y_train, X_test, y_test = load_benchmark_dataset(args.dataset, seed=args.seed)
    n_inputs = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    print(f"\n[1/4] Loaded Dataset '{args.dataset}':")
    print(f"      Input Optical Ports (d) : {n_inputs}")
    print(f"      Output Classes (C)      : {n_classes}")
    print(f"      Streaming Train Samples : {len(X_train)} | Holdout Test Samples: {len(X_test)}")

    # 2. Sweep Photonic MZI Mode Counts
    mode_list = [32, 64, 96, 128, 192, 256]
    if args.modes not in mode_list:
        mode_list.append(args.modes)
        mode_list.sort()

    print(f"\n[2/4] Running Passive MZI Mesh Mode Sweep M in {mode_list}...")
    df_modes = run_hidden_dimension_sweep(X_train, y_train, X_test, y_test, mode_list, seed=args.seed)
    df_modes.to_csv(results_dir / "hidden_dim_sweep.csv", index=False)
    print("\n--- Classification Accuracy vs. Waveguide Modes (M) ---")
    print(
        df_modes[
            [
                "n_modes",
                "n_mzis",
                "optical_delay_ps",
                "photonic_fp64_acc_pct",
                "photonic_q8_16_acc_pct",
                "digital_q8_16_acc_pct",
                "q8_16_degradation_pct",
            ]
        ].to_string(index=False)
    )

    # 3. Thermo-Optic Drift Compensation Benchmark
    print(f"\n[3/4] Evaluating Digital Sherman-Morrison Tracking under SOI Thermo-Optic Drift (M={args.modes})...")
    df_drift = run_thermo_optic_drift_experiment(
        X_train, y_train, X_test, y_test, n_modes=args.modes, seed=args.seed
    )
    df_drift.to_csv(results_dir / "thermal_drift_tracking.csv", index=False)
    final_drift_row = df_drift.iloc[-1]
    mean_adaptive_drift_acc = float(df_drift["adaptive_q8_16_acc_pct"].mean())
    print(
        f"      At Delta_T = +{final_drift_row['delta_T_kelvin']:.2f} K: "
        f"Adaptive Q8.16 OS-ELM = {final_drift_row['adaptive_q8_16_acc_pct']:.2f}% vs. "
        f"Static Readout = {final_drift_row['static_readout_acc_pct']:.2f}% "
        f"(+{final_drift_row['accuracy_gain_pct']:.2f}% recovery)"
    )

    # 4. Electro-Optic Hardware Resource, Latency & Energy Profiling
    print("\n[4/4] Profiling Xilinx Zynq UltraScale+ ZCU104 Hardware Partitioning...")
    profiler = ElectroOpticHardwareProfiler()
    hw_records = []
    for M in [64, 128, 256, 512]:
        mesh_tmp = ClementsMZIMesh(PhotonicMeshConfig(n_modes=M, seed=args.seed))
        hw_records.append(
            profiler.profile_architecture(
                n_inputs=n_inputs,
                n_hidden=M,
                n_classes=n_classes,
                optical_delay_ps=mesh_tmp.optical_propagation_delay_ps,
            )
        )
    df_hw = pd.DataFrame(hw_records)
    df_hw.to_csv(results_dir / "hardware_comparison.csv", index=False)
    print("\n--- Hardware Resource & Latency Comparison (Pure Digital vs. Hybrid Electro-Optic) ---")
    print(
        df_hw[
            [
                "n_hidden",
                "infer_mac_reduction_pct",
                "digital_dsp48e2",
                "hybrid_dsp48e2",
                "digital_infer_latency_us",
                "hybrid_infer_latency_us",
                "latency_speedup_x",
                "energy_reduction_pct",
            ]
        ].to_string(index=False)
    )

    # Generate 300-DPI Figures
    mesh_target = ClementsMZIMesh(PhotonicMeshConfig(n_modes=args.modes, seed=args.seed))
    pd_target = PhotodetectorArray(PhotodetectorConfig(snr_db=34.0, adc_bits=12, seed=args.seed + 59))
    H_sample = pd_target.detect(mesh_target.propagate(X_test[:200]))
    plot_all_figures(results_dir, mesh_target.U_ideal, H_sample, df_modes, df_drift, df_hw)

    # Save master JSON summary for CV & README coherence verification
    row_m = df_modes[df_modes["n_modes"] == args.modes].iloc[0].to_dict()
    hw_m = df_hw[df_hw["n_hidden"] == args.modes].iloc[0].to_dict()
    summary = {
        "project": "Hybrid Photonic-Digital OS-ELM Accelerator (Passive SOI MZI Mesh + Zynq UltraScale+ FPGA)",
        "author": "Harsh Dharmendrabhai Patel",
        "target_modes_M": args.modes,
        "dataset": args.dataset,
        "metrics_at_target_M": {
            "n_mzis": int(row_m["n_mzis"]),
            "optical_propagation_delay_ps": float(row_m["optical_delay_ps"]),
            "unitarity_frobenius_error": float(row_m["unitarity_error_fro"]),
            "photonic_fp64_accuracy_pct": float(row_m["photonic_fp64_acc_pct"]),
            "photonic_q8_16_accuracy_pct": float(row_m["photonic_q8_16_acc_pct"]),
            "digital_q8_16_accuracy_pct": float(row_m["digital_q8_16_acc_pct"]),
            "q8_16_quantization_drop_pct": float(row_m["q8_16_degradation_pct"]),
            "thermo_optic_drift_2_25K_adaptive_acc_pct": float(final_drift_row["adaptive_q8_16_acc_pct"]),
            "thermo_optic_drift_2_25K_static_acc_pct": float(final_drift_row["static_readout_acc_pct"]),
            "thermo_optic_drift_mean_adaptive_acc_pct": round(mean_adaptive_drift_acc, 2),
            "inference_mac_offload_pct": float(hw_m["infer_mac_reduction_pct"]),
            "dsp48e2_reduction_pct": float(hw_m["dsp_reduction_pct"]),
            "digital_infer_latency_us": float(hw_m["digital_infer_latency_us"]),
            "hybrid_infer_latency_us": float(hw_m["hybrid_infer_latency_us"]),
            "latency_speedup_x": float(hw_m["latency_speedup_x"]),
            "dynamic_energy_reduction_pct": float(hw_m["energy_reduction_pct"]),
        },
    }
    with open(results_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Saved CSV tables, JSON summary, and 4 publication figures to: {results_dir}")


if __name__ == "__main__":
    main()
