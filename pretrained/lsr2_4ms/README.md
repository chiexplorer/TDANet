---
license: apache-2.0
language:
- en
tags:
- audio
- audio-source-separation
pipeline_tag: audio-to-audio
---
# An efficient encoder-decoder architecture with top-down attention for speech separation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-efficient-encoder-decoder-architecture/speech-separation-on-libri2mix)](https://paperswithcode.com/sota/speech-separation-on-libri2mix?p=an-efficient-encoder-decoder-architecture) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-efficient-encoder-decoder-architecture/speech-separation-on-wham)](https://paperswithcode.com/sota/speech-separation-on-wham?p=an-efficient-encoder-decoder-architecture)

This repository is the official implementation of [An efficient encoder-decoder architecture with top-down attention for speech separation](https://cslikai.cn/project/TDANet) [Paper link](https://openreview.net/pdf?id=fzberKYWKsI). 

```
@inproceedings{tdanet2023iclr,
  title={An efficient encoder-decoder architecture with top-down attention for speech separation},
  author={Li, Kai and Yang, Runxuan and Hu, Xiaolin},
  booktitle={ICLR},
  year={2023}
}
```

## Training Dataset

- LRS2-2Mix

## Config

```yaml
    enc_kernel_size: 4
    in_channels: 512
    num_blocks: 16
    num_sources: 2
    out_channels: 128
    upsampling_depth: 5
```