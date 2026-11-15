# DNA-CNN

[![CI](https://github.com/kapildhinwa2010-dev/DNA-CNN/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-CNN/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/DNA-CNN/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-CNN/actions/workflows/codeql.yml)


DNA quantized codec tuned for convolutional filters: per-output-channel encoding.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

DNA-CNN is the CNN-specialized member of the DNA codec family
(lineage: the private DNA repository). encode()/decode() quantize a
float matrix into a GeneRecord; the CNN-specific part is
encode_filters()/decode_filters(): a conv weight tensor
(out_ch, in_ch, kh, kw) is encoded per output channel, each (in_ch,
kh, kw) slice with its own scale - lower error than one global scale
when channels have different value ranges.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from dna_cnn import encode_filters, decode_filters

records = encode_filters(conv_weight, bits=8)  # (out_ch, in_ch, kh, kw)
restored = decode_filters(records, conv_weight.shape)
```

## API

- encode(matrix, bits) / decode(record) / codec_stats(matrix, bits)
- encode_filters(filters, bits) - one GeneRecord per output channel
- decode_filters(records, shape) - restores the 4-D tensor

## Honest results and limitations

- Lossy: error equals the quantization step of each channel slice.
- Only 4-D (out_ch, in_ch, kh, kw) conv weights; 1x1 convs may be
  better served by plain encode().
- "Ratio" in codec_stats() is float32-bytes / encoded-bytes
  ARITHMETIC - it counts stored codes, not a real serialized file.
- No per-layer fitting: the reserv.dean-part-2 log records that
  generated-weight reconstructions of trained models were not usable
  without per-layer fitting.

## License

MIT - see [LICENSE](LICENSE).
