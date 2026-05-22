# Self-Supervised Contrastive Learning for Visual Representation

## Overview

This project implements and compares **self-supervised learning (SSL)** methods for visual representation learning. Two frameworks are built from scratch — **SimCLR** and **Barlow Twins** — and trained on 100,000 unlabeled images from the STL-10 dataset. The quality of the learned representations is evaluated by training a logistic regression classifier on top of frozen features, across varying amounts of labeled data (10 to 500 examples per class). A fully supervised baseline trained end-to-end provides a reference point, highlighting the practical advantages of SSL when labels are scarce.

## Dataset

**STL-10** — 96×96 color images across 10 classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck), downscaled to 32×32 for computational efficiency. The dataset provides 100,000 unlabeled images for self-supervised pretraining, 5,000 labeled training images (500 per class), and 8,000 test images (800 per class). Data augmentation for contrastive learning includes random horizontal flips, crop-and-resize, color jitter (brightness/contrast/saturation at 0.5, hue at 0.1), random grayscale, and Gaussian blur.

## Methodology

### Part 1 — SimCLR

SimCLR learns representations by maximizing agreement between two randomly augmented views of the same image via the InfoNCE loss. The architecture consists of a CNN-based encoder (`BaseNetwork`) that produces a representation vector, followed by a two-layer MLP projection head that maps to a 128-dimensional space where the contrastive loss is applied. Training uses AdamW with a cosine annealing schedule, a temperature parameter of 0.07, and a batch size of 256 over 100 epochs.

### Part 2 — Logistic Regression Evaluation

The trained SimCLR encoder is frozen and its output representations are used as fixed features. A single linear layer (logistic regression) is trained on top for classification. To study label efficiency, experiments are run with 10, 20, 50, 100, 200, and 500 labeled examples per class, revealing how well SSL representations transfer under data-scarce conditions.

### Part 3 — Supervised Baseline

A `BaseNetwork` with identical architecture is trained end-to-end from random initialization using all 5,000 labeled training images with cross-entropy loss. Augmentations are adapted for classification (horizontal flip, mild crop-and-resize, grayscale, Gaussian blur — no color distortion). This establishes the performance ceiling of purely supervised learning on this dataset size.

### Part 4 — Barlow Twins

As an alternative SSL approach, Barlow Twins minimizes redundancy between embedding dimensions by pushing the cross-correlation matrix of the two augmented-view embeddings toward the identity matrix. This avoids the need for negative pairs or large batch sizes. The same downstream evaluation pipeline (logistic regression at varying label counts) is applied, and results are compared against SimCLR and the supervised baseline.

## Results

| Method | Labeled Data | Test Accuracy |
|---|---|---|
| SimCLR + Linear | 10 per class | ~34% |
| SimCLR + Linear | 500 per class | ~54% |
| Barlow Twins + Linear | 10–500 per class | Comparable to SimCLR |
| **Supervised Baseline** | **500 per class (all)** | **~60%** |

The supervised baseline achieves the highest accuracy (~60%) when all labeled data is available, which is expected since every parameter is optimized directly for classification. However, it overfits significantly (67% train vs 60% test) due to 234K parameters being fit to only 5,000 examples. SimCLR's linear probe reaches ~54% with 500 labels per class, but the gap narrows as labels decrease — demonstrating that SSL representations generalize better in low-label regimes. The biggest accuracy jump occurs between 10 and 20 labels per class (~7% gain), after which returns diminish logarithmically. Even with just 10 labeled examples per class, the SimCLR linear classifier achieves ~34%, confirming that meaningful visual features are captured without any labels.

## Key Takeaways

- **SSL shines when labels are scarce.** Representations learned from 100K unlabeled images via SimCLR transfer effectively to classification, especially when only a handful of labeled examples are available.
- **The biggest marginal gain comes early.** Going from 10 to 20 labels per class yields ~7% accuracy improvement — far more than any subsequent doubling of data.
- **Supervised training overfits with limited labels.** The baseline's 234K parameters overfit heavily to 5,000 examples (67% train / 60% test), a problem SSL sidesteps by learning representations from unlabeled data.
- **Barlow Twins offers a competitive alternative.** By avoiding explicit negative pairs and instead decorrelating embedding dimensions, Barlow Twins achieves performance comparable to SimCLR while being conceptually simpler.
- **Data augmentation is critical.** The choice and strength of augmentations directly shape what invariances the model learns — this is the single most important hyperparameter in contrastive SSL.

## Tech Stack

- Python, PyTorch, PyTorch Lightning
- torchvision (STL-10 loading and augmentation)
- TensorBoard (training visualization)
- NumPy, Matplotlib, Seaborn

## References

- Chen et al., [*A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*](https://arxiv.org/abs/2002.05709) (ICML 2020)
- Zbontar et al., [*Barlow Twins: Self-Supervised Learning via Redundancy Reduction*](https://arxiv.org/abs/2103.03230) (ICML 2021)
- Oord et al., [*Representation Learning with Contrastive Predictive Coding*](https://arxiv.org/abs/1807.03748) (2018)
- Grill et al., [*Bootstrap Your Own Latent (BYOL)*](https://arxiv.org/abs/2006.07733) (NeurIPS 2020)
