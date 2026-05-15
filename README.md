# SignFlow: End-to-End Sign Language Generation for One-to-Many Modeling using Conditional Flow Matching

Official implementation for the paper **[“SignFlow: End-to-End Sign Language Generation for One-to-Many Modeling using Conditional Flow Matching”](https://dl.acm.org/doi/full/10.1145/3716553.3750765)**.

[![Paper](https://img.shields.io/badge/Paper-ACM%20ICMI%202025-blue)](https://doi.org/10.1145/3716553.3750765)

<!-- Optional badges -->
<!-- [![Paper](https://img.shields.io/badge/Paper-ACM%20ICMI%202025-blue)](#citation) -->
<!-- [![Project Page](https://img.shields.io/badge/Project-Page-green)](#) -->
<!-- [![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license) -->

---

## Introduction

Sign Language Generation (SLG) has received increasing attention in recent years, with various models aiming to produce natural and temporally coherent sign gestures from spoken or written language. However, SLG remains a challenging task due to its inherently one-to-many nature, where a single sentence can correspond to multiple valid gesture sequences, and the requirement for smooth, synchronized motion across multiple articulators. Although diffusion-based models capture diversity, their stochastic denoising often introduces temporal misalignment and motion artifacts. In this work, we propose SignFlow, a novel architecture for SLG based on conditional flow matching with optimal transport. By modeling deterministic flow paths guided by optimal transport and supervised via velocity fields, SignFlow generates gestures that are both semantically coherent and visually smooth. Experiments on the CSL-Daily dataset demonstrate that SignFlow achieves superior BLEU scores and DTW-based motion accuracy compared to both diffusion and autoregressive baselines.

## Environment

## Data

## Models

### Human Models

### Text Encoder

## Training

To train SignFlow, run:

```bash
python train.py --run-name CFM_CSL
```

## Visualization


To visualize generated sign motions, run:

```bash
python vis_csl.py --chkpt-path logs/CFM_CSL/train_step_300000/chkpt.pth
```


## Acknowledgements

We gratefully acknowledge the authors and maintainers of the open-source repositories, datasets, and tools that made this work possible.

Parts of this repository were developed with reference to the following public codebases:

- [SOKE](https://github.com/2000ZRL/SOKE)
- [Conditional Flow Matching](https://github.com/atong01/conditional-flow-matching)
- [SMPL-X](https://github.com/vchoutas/smplx)

We thank the original authors for making their code publicly available. If you use components from these repositories, please also follow their respective licenses and citation requirements.

## License

This repository is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You are free to share this repository for non-commercial purposes with proper attribution. However, modified or adapted versions may not be distributed without permission from the authors.

Please see [license.txt](license.txt) for the full license text.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{khan2025signflow,
  title={SignFlow: End-to-End Sign Language Generation for One-to-Many Modeling using Conditional Flow Matching},
  author={Khan, Nabeela and Wu, Bowen and Tan, Sihan and Ishi, Carlos Toshinori and Nakadai, Kazuhiro},
  booktitle={Proceedings of the 27th International Conference on Multimodal Interaction},
  pages={173--180},
  year={2025}
}
```


THIS REPO IS STILL UNDER CONSTRUCTION.... PLEASE STAY TUNED...