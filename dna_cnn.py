# DNA-CNN - DNA codec tuned for convolutional filter matrices
# Part of the Kapildhinwa2010-dev Toolkit.

"""DNA-Net - encode/decode codec for weight matrices ("genes").

Lineage: the DNA repository. A float matrix is encoded into a compact gene
record (quantized codes + scale metadata) and decoded back; the codec is
deterministic and supports lossless round-trips within quantization error.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "encode", "decode", "GeneRecord", "codec_stats",
    "encode_filters", "decode_filters",
]


class GeneRecord:
    """Compact encoded representation of one weight matrix."""

    __slots__ = ("codes", "lo", "scale", "shape", "bits")

    def __init__(self, codes: np.ndarray, lo: float, scale: float, shape, bits: int) -> None:
        self.codes = codes
        self.lo = lo
        self.scale = scale
        self.shape = shape
        self.bits = bits

    def nbytes(self) -> int:
        return int(self.codes.nbytes + 16)

    def to_dict(self) -> dict:
        return {
            "codes": self.codes,
            "lo": self.lo,
            "scale": self.scale,
            "shape": self.shape,
            "bits": self.bits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneRecord":
        return cls(data["codes"], float(data["lo"]), float(data["scale"]),
                   tuple(data["shape"]), int(data["bits"]))


def encode(matrix: np.ndarray, bits: int = 8) -> GeneRecord:
    """Quantize a matrix into a GeneRecord."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo = float(matrix.min())
    hi = float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    dtype = np.uint16 if bits > 8 else np.uint8
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels).astype(dtype)
    return GeneRecord(codes, lo, scale, matrix.shape, bits)


def decode(record: GeneRecord) -> np.ndarray:
    """Reconstruct the matrix from a GeneRecord."""
    if not isinstance(record, GeneRecord):
        raise TypeError("decode expects a GeneRecord")
    values = np.asarray(record.codes, dtype=np.float64) * record.scale + record.lo
    return values.reshape(record.shape)


def codec_stats(matrix: np.ndarray, bits: int = 8) -> dict:
    """Report compression statistics for encoding ``matrix``."""
    record = encode(matrix, bits=bits)
    original_bytes = matrix.size * 4  # float32
    return {
        "original_bytes": int(original_bytes),
        "encoded_bytes": record.nbytes(),
        "ratio": original_bytes / record.nbytes(),
        "max_abs_error": float(np.abs(matrix - decode(record)).max()),
        "bits": bits,
    }


def encode_filters(filters: np.ndarray, bits: int = 8) -> list:
    """Encode a conv weight tensor (out_ch, in_ch, kh, kw) per output channel.

    Convolution filters of different output channels usually have different
    value ranges. Encoding each (in_ch, kh, kw) slice with its own scale
    gives lower reconstruction error than one global scale.

    Returns: List of GeneRecord, one per output channel.
    """
    filters = np.asarray(filters, dtype=np.float64)
    if filters.ndim != 4:
        raise ValueError("filters must be 4-D (out_ch, in_ch, kh, kw)")
    return [encode(filters[i].reshape(1, -1), bits=bits) for i in range(filters.shape[0])]


def decode_filters(records: list, shape: tuple) -> np.ndarray:
    """Reconstruct a conv weight tensor from encode_filters output.

    shape: original (out_ch, in_ch, kh, kw).
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) != 4 or len(records) != shape[0]:
        raise ValueError("need one record per output channel and a 4-D shape")
    per_channel = shape[1] * shape[2] * shape[3]
    slices = [decode(r).reshape(per_channel) for r in records]
    return np.stack(slices, axis=0).reshape(shape)