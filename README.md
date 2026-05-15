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


## Visualization


## Acknowledgements


## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{khan2025signflow,
  title     = {SignFlow: End-to-End Sign Language Generation for One-to-Many Modeling using Conditional Flow Matching},
  author    = {Khan, Nabeela and others},
  booktitle = {Proceedings of the ACM International Conference on Multimodal Interaction},
  year      = {2025}
}
```


CODE COMING SOON....