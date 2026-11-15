import numpy as np
import pytest

from dna_cnn import encode_filters, decode_filters


def _filters(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((4, 3, 3, 3))


def test_filters_roundtrip():
    F = _filters()
    records = encode_filters(F, bits=8)
    assert len(records) == 4
    out = decode_filters(records, F.shape)
    assert out.shape == F.shape
    assert np.abs(F - out).max() < 0.05


def test_filters_per_channel_beats_global_on_mixed_ranges():
    rng = np.random.default_rng(1)
    chans = [rng.standard_normal((3, 3, 3)) * s for s in (0.01, 100.0)]
    F = np.stack(chans, axis=0)
    out = decode_filters(encode_filters(F, bits=8), F.shape)
    # Large channel must be reconstructed with small relative error
    rel = np.abs(F[1] - out[1]).max() / 100.0
    assert rel < 0.05


def test_filters_rejects_2d():
    with pytest.raises(ValueError):
        encode_filters(np.zeros((4, 4)))


def test_decode_filters_rejects_bad_shape():
    F = _filters()
    records = encode_filters(F, bits=8)
    with pytest.raises(ValueError):
        decode_filters(records, (2, 3, 3, 3))
