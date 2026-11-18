"""
Digital FPGA Backend: Sequential Sherman-Morrison OS-ELM Readout
================================================================
Implements the Online Sequential Extreme Learning Machine (OS-ELM) adaptive
readout layer using rank-1 Sherman-Morrison recursive updates (Liang et al.,
IEEE TNN 2006) in both 64-bit floating point (FP64) and 24-bit signed fixed-point
(Q8.16 / ap_fixed<24,8>) arithmetic matching a Xilinx Zynq UltraScale+ FPGA.

Hardware Numerical Stability Features:
- Symmetric Covariance Enforcement: P_k <- 0.5 * (P_k + P_k^T) to eliminate
  finite-word-length asymmetry accumulation in 24-bit Q8.16 arithmetic.
- Covariance Windup Prevention: Stabilized forgetting factor update with diagonal
  Levenberg-Marquardt leakage towards P_0 to guarantee positive-definiteness and
  prevent Q8.16 saturation during continuous multi-thousand-sample streaming.
"""

import numpy as np
from .fixed_point import FixedPointQ8_16


class ShermanMorrisonReadout:
    """
    Adaptive digital OS-ELM readout using rank-1 Sherman-Morrison recursive updates.
    Supports both FP64 software reference and cycle-accurate Q8.16 fixed-point emulation.
    """

    def __init__(
        self,
        n_hidden: int,
        n_classes: int,
        reg_lambda: float = 2.5,
        forgetting_factor: float = 1.0,
        use_fixed_point: bool = False,
    ):
        self.L = n_hidden
        self.C = n_classes
        self.reg_lambda = reg_lambda
        self.forgetting_factor = forgetting_factor
        self.use_fixed_point = use_fixed_point

        p_init = (1.0 / reg_lambda) * np.eye(self.L, dtype=np.float64)
        beta_init = np.zeros((self.L, self.C), dtype=np.float64)

        if self.use_fixed_point:
            self.P_0_diag = FixedPointQ8_16.quantize(1.0 / reg_lambda)
            self.P = FixedPointQ8_16.quantize(p_init)
            self.beta = FixedPointQ8_16.quantize(beta_init)
        else:
            self.P_0_diag = 1.0 / reg_lambda
            self.P = p_init
            self.beta = beta_init

        self.samples_seen = 0

    @staticmethod
    def _to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
        """Convert integer class labels into [-1, +1] target encoding for balanced Q8.16 swing."""
        y = np.atleast_1d(np.asarray(y, dtype=int))
        Y = -1.0 * np.ones((len(y), n_classes), dtype=np.float64)
        Y[np.arange(len(y)), y] = 1.0
        return Y

    def partial_fit_sample(self, h_k: np.ndarray, y_vec: np.ndarray) -> None:
        """
        Perform a single sample rank-1 Sherman-Morrison recursive update.
        Avoids all matrix inversions (O(L^3)) in favor of O(L^2) MAC operations.
        """
        lam = self.forgetting_factor

        if not self.use_fixed_point:
            g = self.P @ h_k
            h_dot_g = max(0.0, float(h_k @ g))
            gamma = lam + h_dot_g
            k = g / gamma
            err = y_vec - (h_k @ self.beta)
            self.beta = self.beta + np.outer(k, err)
            p_next = (self.P - np.outer(k, g)) / lam
            if lam < 1.0:
                leak = 1.0 - lam
                p_next = (1.0 - leak) * p_next + leak * self.P_0_diag * np.eye(self.L)
            self.P = 0.5 * (p_next + p_next.T)
        else:
            fp = FixedPointQ8_16
            h_q = fp.quantize(h_k)
            y_q = fp.quantize(y_vec)

            # Step 1: g_k = P_{k-1} * h_k
            g_q = fp.matvec(self.P, h_q)

            # Step 2: gamma_k = lambda + max(0, h_k^T * g_k)
            h_dot_g = max(0.0, fp.quantize(float(np.dot(h_q, g_q))))
            gamma_q = fp.quantize(lam + h_dot_g)

            # Step 3: Kalman gain k_k = g_k / gamma_k
            k_q = fp.div_scalar(g_q, gamma_q)

            # Step 4: Innovation error e_k = y_k - h_k^T * beta_{k-1}
            pred_q = fp.matvec(self.beta.T, h_q)
            err_q = fp.sub(y_q, pred_q)

            # Step 5: Readout update beta_k = beta_{k-1} + k_k * e_k^T
            delta_beta = fp.outer(k_q, err_q)
            self.beta = fp.add(self.beta, delta_beta)

            # Step 6: Sherman-Morrison rank-1 covariance update with symmetry & anti-windup
            rank1_update = fp.outer(k_q, g_q)
            p_diff = fp.sub(self.P, rank1_update)
            if abs(lam - 1.0) > 1e-6:
                leak = 1.0 - lam
                p_scaled = p_diff / lam
                p_next = (1.0 - leak) * p_scaled + leak * self.P_0_diag * np.eye(self.L)
            else:
                p_next = p_diff
            self.P = fp.quantize(0.5 * (p_next + p_next.T))

        self.samples_seen += 1

    def partial_fit(self, H: np.ndarray, y: np.ndarray) -> None:
        """Sequentially update the readout layer sample-by-sample across batch H."""
        H = np.atleast_2d(np.asarray(H, dtype=np.float64))
        Y = self._to_one_hot(y, self.C)
        for i in range(len(H)):
            self.partial_fit_sample(H[i], Y[i])

    def predict_scores(self, H: np.ndarray) -> np.ndarray:
        """Compute raw readout scores H @ beta."""
        H = np.atleast_2d(np.asarray(H, dtype=np.float64))
        if not self.use_fixed_point:
            return H @ self.beta
        fp = FixedPointQ8_16
        H_q = fp.quantize(H)
        return fp.quantize(H_q @ self.beta)

    def predict(self, H: np.ndarray) -> np.ndarray:
        """Predict integer class indices argmax(H @ beta)."""
        scores = self.predict_scores(H)
        return np.argmax(scores, axis=1)

    def score(self, H: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy in [0, 1]."""
        preds = self.predict(H)
        return float(np.mean(preds == np.asarray(y, dtype=int)))


class DigitalRandomHiddenLayer:
    """
    Standard Digital OS-ELM random hidden layer baseline for hardware comparison.
    Stores a dense d x L random weight matrix W_rand in FPGA BRAM and computes
    H = Sigmoid(X @ W_rand + b) using FPGA DSP48E2 multipliers.
    """

    def __init__(self, n_inputs: int, n_hidden: int, seed: int = 42, use_fixed_point: bool = False):
        self.d = n_inputs
        self.L = n_hidden
        self.use_fixed_point = use_fixed_point
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_inputs), size=(n_inputs, n_hidden))
        self.b = rng.uniform(-0.25, 0.25, size=(1, n_hidden))
        if use_fixed_point:
            self.W = FixedPointQ8_16.quantize(self.W)
            self.b = FixedPointQ8_16.quantize(self.b)

    def project(self, X: np.ndarray) -> np.ndarray:
        """Compute digital random projection with sigmoid activation and L2 scaling."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        x_min = X.min(axis=1, keepdims=True)
        x_ptp = np.ptp(X, axis=1, keepdims=True) + 1e-8
        X_norm = (X - x_min) / x_ptp
        if self.use_fixed_point:
            X_norm = FixedPointQ8_16.quantize(X_norm)
            pre = FixedPointQ8_16.quantize(X_norm @ self.W + self.b)
            act = 1.0 / (1.0 + np.exp(-np.clip(pre * 2.5, -10.0, 10.0)))
            l2_norm = np.linalg.norm(act, axis=1, keepdims=True) + 1e-9
            return FixedPointQ8_16.quantize((act / l2_norm) * 4.0)
        pre = X_norm @ self.W + self.b
        act = 1.0 / (1.0 + np.exp(-np.clip(pre * 2.5, -10.0, 10.0)))
        l2_norm = np.linalg.norm(act, axis=1, keepdims=True) + 1e-9
        return (act / l2_norm) * 4.0
