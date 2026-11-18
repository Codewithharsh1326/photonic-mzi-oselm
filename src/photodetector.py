"""
Germanium-on-Silicon (Ge-on-Si) Photodetector & TIA/ADC Readout Array
=====================================================================
Models the optoelectronic boundary converting coherent optical fields E_out from
the passive MZI mesh into digitized intensity features for the FPGA backend.

Physical Nonlinearity:
    I_j = R_PD * |E_out,j|^2 + i_shot + i_thermal
The intrinsic square-law modulus-squared response of high-speed Ge photodiodes
provides the required nonlinear activation function in the optical domain with
zero digital LUT overhead.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PhotodetectorConfig:
    """Parameters for Ge-on-Si waveguide photodetectors and high-speed TIAs/ADCs."""
    responsivity_a_per_w: float = 0.95      # Ge-on-Si photodiode responsivity at 1550 nm (A/W)
    bandwidth_ghz: float = 20.0             # Optoelectronic receiver bandwidth (GHz)
    snr_db: float = 34.0                    # Effective optoelectronic SNR (shot + TIA thermal noise)
    adc_bits: int = 12                      # High-speed ADC resolution before Q8.16 mapping
    seed: int = 101                         # Noise generator seed


class PhotodetectorArray:
    """Simulates square-law intensity detection, optoelectronic noise, and ADC quantization."""

    def __init__(self, config: PhotodetectorConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def detect(self, E_out: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Convert complex optical output fields E_out (N_samples, M) into normalized
        electrical hidden-layer feature vectors H (N_samples, M) scaled for Q8.16 stability.
        """
        intensity = self.cfg.responsivity_a_per_w * (np.abs(E_out) ** 2)

        if add_noise and self.cfg.snr_db is not None:
            sig_power = np.mean(intensity ** 2, axis=1, keepdims=True) + 1e-12
            noise_std = np.sqrt(sig_power / (10.0 ** (self.cfg.snr_db / 10.0)))
            shot_profile = 0.5 * (np.sqrt(intensity / (np.mean(intensity) + 1e-12)) + 1.0)
            noise = self.rng.normal(0.0, 1.0, size=intensity.shape) * noise_std * shot_profile
            intensity = np.maximum(0.0, intensity + noise)

        # TIA automatic gain control (AGC): scale L2 norm to O(1) so h^T P h stays bounded in Q8.16
        l2_norm = np.linalg.norm(intensity, axis=1, keepdims=True) + 1e-9
        h_analog = (intensity / l2_norm) * 4.0

        if self.cfg.adc_bits is not None and self.cfg.adc_bits > 0:
            levels = (1 << self.cfg.adc_bits) - 1
            fs_max = 2.5
            h_clipped = np.clip(h_analog, 0.0, fs_max)
            h_digital = np.rint((h_clipped / fs_max) * levels) * (fs_max / levels)
            return h_digital

        return h_analog
