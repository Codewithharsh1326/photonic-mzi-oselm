"""
24-Bit Fixed-Point (Q8.16 / ap_fixed<24,8>) Datapath Emulator
=============================================================
Emulates the custom 24-bit signed fixed-point digital hardware datapath on a
Xilinx Zynq UltraScale+ (ZCU104) FPGA used for sequential Sherman-Morrison
readout updates.

Format Specification:
- Total bits (W): 24 bits (signed two's complement)
- Integer bits (I): 8 bits (1 sign bit + 7 magnitude bits, range [-128.0, +127.9999847])
- Fractional bits (F): 16 bits (resolution 2^-16 = 1.5258789e-5)
"""

import numpy as np


class FixedPointQ8_16:
    """Emulator for signed 24-bit Q8.16 fixed-point arithmetic (Vitis HLS ap_fixed<24,8>)."""

    TOTAL_BITS = 24
    INT_BITS = 8
    FRAC_BITS = 16
    SCALE = 1 << FRAC_BITS  # 65536
    MIN_INT = -(1 << (TOTAL_BITS - 1))       # -8,388,608
    MAX_INT = (1 << (TOTAL_BITS - 1)) - 1    # +8,388,607
    MIN_VAL = MIN_INT / SCALE                # -128.0
    MAX_VAL = MAX_INT / SCALE                # +127.99998474121094
    LSB = 1.0 / SCALE

    @classmethod
    def quantize(cls, x: np.ndarray | float) -> np.ndarray | float:
        """Quantize floating-point array or scalar to signed Q8.16 with saturation."""
        scaled = np.rint(np.asarray(x, dtype=np.float64) * cls.SCALE)
        clipped = np.clip(scaled, cls.MIN_INT, cls.MAX_INT)
        res = clipped / cls.SCALE
        if np.isscalar(x):
            return float(res)
        return res

    @classmethod
    def add(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Saturating Q8.16 addition."""
        return cls.quantize(cls.quantize(a) + cls.quantize(b))

    @classmethod
    def sub(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Saturating Q8.16 subtraction."""
        return cls.quantize(cls.quantize(a) - cls.quantize(b))

    @classmethod
    def mul(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """48-bit intermediate product truncated/rounded back to Q8.16 with saturation."""
        return cls.quantize(cls.quantize(a) * cls.quantize(b))

    @classmethod
    def matvec(cls, mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Fixed-point matrix-vector product with 48-bit accumulator and Q8.16 output."""
        q_mat = cls.quantize(mat)
        q_vec = cls.quantize(vec)
        acc = q_mat @ q_vec
        return cls.quantize(acc)

    @classmethod
    def outer(cls, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Fixed-point rank-1 outer product u * v^T quantized to Q8.16."""
        q_u = cls.quantize(u)
        q_v = cls.quantize(v)
        return cls.quantize(np.outer(q_u, q_v))

    @classmethod
    def div_scalar(cls, num: np.ndarray, den: float) -> np.ndarray:
        """Fixed-point reciprocal-multiply division (LUT + 1 Newton-Raphson step on FPGA)."""
        q_den = max(abs(cls.quantize(den)), cls.LSB)
        if den < 0:
            q_den = -q_den
        inv_den = cls.quantize(1.0 / q_den)
        return cls.quantize(cls.quantize(num) * inv_den)
