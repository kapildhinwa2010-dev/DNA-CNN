# DNA-CNN

DNA-CNN is a small NumPy-based, lossy quantization codec for two-dimensional weight matrices and four-dimensional convolution-weight arrays. Its CNN-specific operation quantizes each output-channel slice independently; this repository does not implement a CNN model, training loop, or inference pipeline. The code is a compact utility/prototype rather than a production-ready model-compression system.

## What is implemented

- `encode(matrix, bits=8)` maps a non-empty 2-D array to a `GeneRecord` using min–max linear quantization. The supported bit setting is 1 through 16.
- `decode(record)` reconstructs a matrix from its codes and scale metadata.
- `encode_filters(filters, bits=8)` accepts a 4-D array with shape `(out_ch, in_ch, kh, kw)` and returns one `GeneRecord` per output channel.
- `decode_filters(records, shape)` reconstructs a 4-D convolution-weight array from those records.
- `GeneRecord.to_dict()` and `GeneRecord.from_dict()` expose a basic dictionary conversion; `codec_stats()` reports an arithmetic size estimate and maximum absolute reconstruction error.

The API is NumPy-based. Filter encoding applies the matrix encoder separately to each output-channel slice; it does not alter or compress a complete neural-network model.

## Install

The project declares Python 3.9 or newer and NumPy 1.24 or newer. From the repository root:

```bash
python -m pip install .
```

To install the dependencies used by the test suite instead:

```bash
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np
from dna_cnn import encode_filters, decode_filters

weights = np.random.default_rng(0).standard_normal((16, 3, 3, 3))
records = encode_filters(weights, bits=8)
restored = decode_filters(records, weights.shape)
```

For a matrix, use `encode(matrix, bits=8)` and `decode(record)`. `codec_stats(matrix, bits=8)` returns `original_bytes`, `encoded_bytes`, `ratio`, `max_abs_error`, and `bits`.

## Tests and CI

Run the tests with:

```bash
python -m pytest -q
```

The suite was run against commit `8c1232367ca6366098c574617731eb4b562f0abd`: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider` completed with **16 passed**. The GitHub Actions CI workflow is configured to run pytest, compile the module and tests, and perform a basic import/encode/decode smoke check on Python 3.10, 3.11, and 3.12. This statement describes the workflow configuration; it is not a claim about the current remote workflow status.

## Limitations

- Encoding is lossy scalar min–max quantization; no learned codebook, per-layer calibration, or model-aware fitting is implemented.
- Filter-specific handling is limited to independent quantization of output-channel slices in 4-D arrays. There is no framework adapter, model loader, training/inference code, dataset, or documented end-to-end model-compression evaluation in this repository.
- Codes are stored in NumPy integer arrays (`uint8` for up to 8 bits, `uint16` above 8 bits); values above 8 bits are not tightly bit-packed.
- `GeneRecord` is an in-memory Python object. The dictionary conversion is not a file serialization format. `codec_stats()` estimates encoded bytes as code-array bytes plus 16 bytes of metadata and compares that with a float32-size baseline; it does not measure a serialized artifact or actual end-to-end storage savings.
- Reconstruction quality depends on the input range and chosen bit depth. The tests exercise selected synthetic inputs and do not establish general quality or performance guarantees.

## License

The repository includes an MIT License; see [LICENSE](LICENSE).
