"""
Passive Silicon Photonic Mach-Zehnder Interferometer (MZI) Mesh Simulator
=========================================================================
Models an untuned, passive M-mode rectangular multiport interferometer following
the Clements topology (Clements et al., Optica 2016) fabricated on a 220 nm
Silicon-on-Insulator (SOI) platform operating at telecom wavelength lambda = 1550 nm.

Key Physical Principles:
1. Zero-Power Random Unitary Projection:
   Instead of actively tuning N_MZI = M(M-1)/2 thermo-optic phase shifters (~18 mW/heater)
   to program deterministic weights via SVD, an untuned passive MZI mesh exploits random
   static fabrication phase offsets (theta, phi ~ U(0, 2*pi)) arising from nanometer-scale
   SOI waveguide width variations. This realizes a dense, complex unitary random projection
   at the speed of light (~32 ps propagation delay) with 0 mW static tuning power.
2. Thermo-Optic Phase Drift Modeling:
   Models global/gradient chip temperature drift Delta_T (K) via the silicon thermo-optic
   coefficient dn_eff/dT = 1.86e-4 K^-1 at lambda = 1550 nm, enabling evaluation of
   digital Sherman-Morrison OS-ELM adaptive drift compensation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PhotonicMeshConfig:
    """Physical and architectural parameters for the passive SOI Clements MZI mesh."""
    n_modes: int = 128                      # Number of optical waveguide modes (M)
    wavelength_nm: float = 1550.0           # Operating telecom wavelength (nm)
    n_group: float = 4.20                   # SOI strip waveguide group index (n_g)
    dn_eff_dT: float = 1.86e-4              # Silicon thermo-optic coefficient (K^-1)
    mzi_length_um: float = 18.0             # Compact passive directional-coupler MZI cell pitch (um)
    phase_arm_length_um: float = 45.0       # Effective unbalanced optical path length (um)
    loss_db_per_layer: float = 0.035        # Insertion loss per MZI layer (dB)
    seed: int = 42                          # Deterministic fabrication seed


class ClementsMZIMesh:
    """
    Simulates an untuned M-mode Clements rectangular Mach-Zehnder Interferometer mesh.

    Each 2x2 MZI consists of two 50:50 directional couplers (or MMIs) and static internal
    (theta) and external (phi) phase shifts:
        T(theta, phi) = i * exp(i * theta / 2) *
            [[exp(i * phi) * sin(theta / 2),  cos(theta / 2)],
             [exp(i * phi) * cos(theta / 2), -sin(theta / 2)]]
    """

    def __init__(self, config: PhotonicMeshConfig):
        self.cfg = config
        self.M = config.n_modes
        self.n_layers = config.n_modes
        self.n_mzis = (self.M * (self.M - 1)) // 2
        self.rng = np.random.default_rng(config.seed)

        # Static untuned fabrication phase offsets ~ U(0, 2*pi)
        self.theta_0 = self.rng.uniform(0.0, 2.0 * np.pi, size=(self.n_layers, self.M // 2))
        self.phi_0 = self.rng.uniform(0.0, 2.0 * np.pi, size=(self.n_layers, self.M // 2))

        # Per-MZI spatial sensitivity factor (models spatial thermal gradient across the PIC)
        self.thermal_sensitivity = self.rng.uniform(0.75, 1.25, size=(self.n_layers, self.M // 2))

        # Input embedding random fixed phase mask for coherent dual-rail encoding
        self.input_phase_mask = np.exp(1j * self.rng.uniform(0.0, 2.0 * np.pi, size=self.M))

        # Assemble nominal unitary transfer matrix U_0 (at Delta_T = 0 K)
        self.U_ideal = self._assemble_unitary(delta_T_kelvin=0.0)
        self.field_loss_factor = 10.0 ** (-0.05 * self.cfg.loss_db_per_layer * self.n_layers)
        self.U_effective = self.field_loss_factor * self.U_ideal

    @property
    def chip_length_mm(self) -> float:
        """Total physical length of the M-layer Clements mesh in millimeters."""
        return (self.n_layers * self.cfg.mzi_length_um) * 1e-3

    @property
    def optical_propagation_delay_ps(self) -> float:
        """Group delay (time-of-flight) of light through the passive MZI mesh in picoseconds."""
        c_m_per_s = 299_792_458.0
        length_m = self.chip_length_mm * 1e-3
        delay_s = (length_m * self.cfg.n_group) / c_m_per_s
        return delay_s * 1e12

    @property
    def total_insertion_loss_db(self) -> float:
        """Total optical insertion loss across all M layers in dB."""
        return self.n_layers * self.cfg.loss_db_per_layer

    def _assemble_unitary(self, delta_T_kelvin: float = 0.0) -> np.ndarray:
        """
        Construct the M x M unitary matrix U by cascading M layers of 2x2 MZIs
        in the rectangular Clements arrangement.
        """
        lambda_um = self.cfg.wavelength_nm * 1e-3
        base_dphi = (
            (2.0 * np.pi * self.cfg.phase_arm_length_um / lambda_um)
            * self.cfg.dn_eff_dT
            * delta_T_kelvin
        )

        theta = self.theta_0 + base_dphi * self.thermal_sensitivity
        phi = self.phi_0 + 0.65 * base_dphi * self.thermal_sensitivity

        U = np.eye(self.M, dtype=np.complex128)

        for layer in range(self.n_layers):
            start_mode = layer % 2
            n_pairs = (self.M - start_mode) // 2

            idx1 = start_mode + 2 * np.arange(n_pairs)
            idx2 = idx1 + 1

            th = theta[layer, :n_pairs]
            ph = phi[layer, :n_pairs]

            half_th = 0.5 * th
            prefactor = 1j * np.exp(1j * half_th)
            exp_ph = np.exp(1j * ph)
            sin_th = np.sin(half_th)
            cos_th = np.cos(half_th)

            t11 = (prefactor * exp_ph * sin_th)[:, None]
            t12 = (prefactor * cos_th)[:, None]
            t21 = (prefactor * exp_ph * cos_th)[:, None]
            t22 = (-prefactor * sin_th)[:, None]

            u1 = U[idx1, :].copy()
            u2 = U[idx2, :].copy()

            U[idx1, :] = t11 * u1 + t12 * u2
            U[idx2, :] = t21 * u1 + t22 * u2

        return U

    def verify_unitarity_error(self) -> float:
        """Compute Frobenius norm error ||U_ideal^H U_ideal - I||_F to verify exact unitarity."""
        ident = np.eye(self.M, dtype=np.complex128)
        return float(np.linalg.norm(self.U_ideal.conj().T @ self.U_ideal - ident, ord="fro"))

    def encode_optical_field(self, X: np.ndarray, optical_power_mw: float = 10.0) -> np.ndarray:
        """
        Encode real input vectors X (N_samples, d) into M-mode coherent optical fields E_in.
        Uses dual-rail amplitude + phase modulation with a coherent reference bias mode so that
        interferometric mixing converts both amplitude and phase information into output intensity.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples, d = X.shape

        x_min = X.min(axis=1, keepdims=True)
        x_ptp = np.ptp(X, axis=1, keepdims=True) + 1e-8
        X_norm = (X - x_min) / x_ptp

        E_in = np.zeros((n_samples, self.M), dtype=np.complex128)
        for m in range(self.M):
            feat_idx = m % d
            band = m // d
            val = X_norm[:, feat_idx]
            if band % 2 == 0:
                E_in[:, m] = (0.35 + 0.65 * val) * np.exp(1j * (0.5 * np.pi * val))
            else:
                E_in[:, m] = np.sqrt(1.0 - 0.5 * val**2) * np.exp(1j * (np.pi * val + 0.25 * np.pi * band))

        E_in = E_in * self.input_phase_mask[None, :]
        norms = np.linalg.norm(E_in, axis=1, keepdims=True) + 1e-12
        E_in = E_in / norms * np.sqrt(optical_power_mw)
        return E_in

    def propagate(
        self,
        X: np.ndarray,
        optical_power_mw: float = 10.0,
        delta_T_kelvin: float = 0.0,
    ) -> np.ndarray:
        """
        Propagate input samples X through the passive MZI mesh:
            E_out = U_effective(Delta_T) @ E_in^T
        Returns complex optical output fields of shape (N_samples, M).
        """
        E_in = self.encode_optical_field(X, optical_power_mw=optical_power_mw)
        if abs(delta_T_kelvin) < 1e-9:
            U_eff = self.U_effective
        else:
            U_eff = self.field_loss_factor * self._assemble_unitary(delta_T_kelvin=delta_T_kelvin)
        return E_in @ U_eff.T
