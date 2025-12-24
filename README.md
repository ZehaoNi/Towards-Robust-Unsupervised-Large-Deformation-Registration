# Towards-Robust-Unsupervised-Large-Deformation-Registration

This is the official implementation of the paper "Towards Robust Unsupervised Large Deformation Registration for Prostate MRI".


🏆 Highlights
Adaptive Dual-Enhanced Feature Fusion (ADFF): A transformer-convnet hybrid that simultaneously captures global dependencies and local details.

RIN/RN Module: Enhances feature extraction for complex anatomical boundaries.

Uncertainty-Aware Regularization (UAR): A plug-and-play module that enforces deformation field consistency, significantly boosting robustness with negligible computational overhead.


🚀 Quick Start
1. Prerequisites
Python >= 3.8

PyTorch >= 1.10 (Recommended: 2.0+ with CUDA 11.3)

A modern NVIDIA GPU (≥ 12GB VRAM recommended for full training, e.g., RTX 3090/A100)

2. Installation

Key dependencies (requirements.txt):
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
SimpleITK>=2.2.0
nibabel>=5.0.0
scikit-image>=0.20.0
tqdm>=4.65.0
matplotlib>=3.7.0
pyyaml>=6.0

3. Data Preparation
a) Download Datasets
MSD Prostate Task05: Download from the Medical Segmentation Decathlon.

µ-RegPro: Available via the MICCAI 2023 Prostate MR-US Registration Challenge.

HPH-FU: (Internal dataset, contact authors for access information).

b) Preprocessing
Run our preprocessing script to standardize all data:
The script performs:

Resampling to isotropic resolution.
Cropping/padding to a uniform size of 160×160×96.
Intensity normalization (Z-score).
Optional data augmentation (random flipping) for training.
