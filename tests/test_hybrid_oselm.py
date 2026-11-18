"""
Unit Tests for Hybrid Photonic-Digital OS-ELM Co-Design Simulator
=================================================================
Run via:
    python3 -m unittest discover -s tests
"""

import unittest
import numpy as np

from src.mzi_mesh import PhotonicMeshConfig, ClementsMZIMesh
from src.photodetector import PhotodetectorConfig, PhotodetectorArray
from src.fixed_point import FixedPointQ8_16
from src.digital_oselm import ShermanMorrisonReadout
from src.dataset_loader import load_benchmark_dataset


class TestHybridPhotonicOSELM(unittest.TestCase):
    """Verification suite for optical unitarity, Q8.16 arithmetic, and OS-ELM convergence."""

    def test_clements_mzi_unitarity(self):
        """Verify that the lossless M-mode Clements MZI mesh transfer matrix is strictly unitary."""
        cfg = PhotonicMeshConfig(n_modes=64, loss_db_per_layer=0.0, seed=42)
        mesh = ClementsMZIMesh(cfg)
        err = mesh.verify_unitarity_error()
        self.assertLess(err, 1e-12, f"Unitarity error too high: {err}")

    def test_fixed_point_q8_16_bounds_and_resolution(self):
        """Verify 24-bit Q8.16 saturation and quantization LSB step."""
        self.assertAlmostEqual(FixedPointQ8_16.quantize(200.0), FixedPointQ8_16.MAX_VAL, places=5)
        self.assertAlmostEqual(FixedPointQ8_16.quantize(-200.0), FixedPointQ8_16.MIN_VAL, places=5)
        val = 1.23456
        q_val = FixedPointQ8_16.quantize(val)
        self.assertLessEqual(abs(val - q_val), 0.5 * FixedPointQ8_16.LSB + 1e-9)

    def test_hybrid_photonic_oselm_accuracy(self):
        """Verify that 128-mode Photonic MZI + Q8.16 Sherman-Morrison readout achieves >95% accuracy."""
        X_train, y_train, X_test, y_test = load_benchmark_dataset("optical_digits", seed=42)
        mesh = ClementsMZIMesh(PhotonicMeshConfig(n_modes=128, seed=42))
        pd_array = PhotodetectorArray(PhotodetectorConfig(snr_db=38.0, adc_bits=12, seed=101))

        H_train = pd_array.detect(mesh.propagate(X_train))
        H_test = pd_array.detect(mesh.propagate(X_test))

        readout_q = ShermanMorrisonReadout(n_hidden=128, n_classes=10, reg_lambda=2.0, use_fixed_point=True)
        readout_q.partial_fit(H_train, y_train)
        acc = readout_q.score(H_test, y_test)
        self.assertGreater(acc, 0.95, f"Expected >95% test accuracy, got {acc * 100:.2f}%")


if __name__ == "__main__":
    unittest.main()
