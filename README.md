# Towards-Robust-Unsupervised-Large-Deformation-Registration

ARE-Net is an advanced unsupervised volumetric medical image registration network designed for large deformation prostate MRI. It introduces an adaptive robust dual-enhanced architecture that effectively integrates global contextual information with local feature details while ensuring high registration consistency through uncertainty-aware regularization.

🖥️ Experimental Setup
Hardware Configuration
All experiments were conducted on a high-performance computing workstation equipped with:

GPU: NVIDIA Tesla A100 (40GB VRAM) for primary experiments

Alternative GPU: NVIDIA GeForce RTX 3080/3090 (≥12GB VRAM) for development

CPU: Intel Xeon Gold processors

RAM: 64GB minimum (recommended 128GB for large datasets)

Storage: NVMe SSD for fast data loading during training

Software Environment
Our implementation was developed on Ubuntu 20.04 LTS using:

Python: 3.8+

PyTorch: 2.0+ with CUDA 11.8

Key Dependencies:

torch, torchvision (for deep learning framework)

nibabel, SimpleITK (for medical image I/O)

numpy, scipy (for numerical operations)

scikit-image, opencv-python (for image processing)

tqdm, matplotlib (for progress tracking and visualization)

pyyaml (for configuration management)



🏗️ Method Overview
The core of our proposed ARE-Net is an adaptive robust dual-enhanced architecture with three innovative components. The following diagram illustrates the overall framework:




1. Adaptive Dual-Enhanced Feature Fusion (ADFF) Framework
The input image pair is processed through a parallel architecture:
Transformer Branch: Based on 3D Swin Transformer blocks with multi-head self-attention to capture long-range spatial dependencies and global contextual information across four downsampling stages (rates: 4, 2, 2, 2).
ConvNet Branch: Built with our enhanced RIN/RN modules (ResInceptNeXt/ResNeXt) for extracting hierarchical local features across five downsampling stages (rate: 2 at each stage).


2. RIN/RN Module for Enhanced Feature Extraction
RIN Module (Residual Inception ConvNeXt): Captures multi-scale information using depth-wise separable convolutions with kernel dimensions of 3×3×3, 5×5×5, and 7×7×7, followed by GroupNorm and GELU activation, with residual connections to facilitate gradient flow.

RN Module (Residual ConvNeXt): Performs efficient downsampling while preserving semantic richness through separable convolutions, stride convolutions (stride=2), and expansion layers with residual connections.

3. Uncertainty-Aware Regularization (UAR) Module
A plug-and-play component that enhances deformation field consistency:

Stochastic Deformation Generation: Uses Monte Carlo dropout to generate multiple deformation field samples during forward propagation.

Uncertainty Quantification: Computes voxel-wise variance across deformation field ensembles to quantify predictive uncertainty.

Robustness Optimization: Applies uncertainty-aware loss to penalize prediction variance, enforcing solution consistency without additional computational overhead.



📊 Datasets
The model was trained and evaluated on three volumetric medical image datasets:

HPH-FU (Internal Prostate MRI Follow-up)
Description: 90 T2-weighted MRI pairs collected from prostate patients at two different time points, reflecting real-world clinical follow-up scenarios with significant physiological deformations.

Split: 168 volumes for training, 72 for testing (after augmentation to 270 pairs)

Annotations: Expert-annotated masks for central gland and peripheral zone by radiologists with >10 years experience

MSD Prostate Task05 (Public Prostate MRI)
Description: 32 T2-weighted MRI volumes from the Medical Segmentation Decathlon, each annotated for central gland and peripheral zone.

Split: 168 pairs for training, 72 for testing (after augmentation)

Purpose: Validation of generalization capability on standardized public benchmark

µ-RegPro (Public Multimodal Prostate)
Description: Multimodal dataset from the MICCAI Prostate MR-US Registration Challenge, containing over 100 paired MR and TRUS images.

Split: 73 image pairs for training, 35 for testing

Purpose: Evaluation of cross-modal registration performance between MRI and ultrasound

🔄 Data Preprocessing
All volumetric data underwent a consistent preprocessing pipeline implemented in preprocess.py:

Anatomical Focus: Retained only the prostate region framework to ensure registration focuses on target anatomy.

Spatial Standardization: All data were preprocessed to the same spatial coordinate system and resampled to uniform voxel spacing.

Dimension Unification: Images were resized to fixed dimensions of 160×160×96 for consistent model processing.

Intensity Normalization: Image intensities were normalized using Z-score normalization to reduce variations from different scanning devices or protocols.

Training Augmentation: For training data only, random flipping was applied to artificially expand the dataset and improve model robustness.

Resampling to isotropic resolution.
Cropping/padding to a uniform size of 160×160×96.
Intensity normalization (Z-score).
Optional data augmentation (random flipping) for training.
