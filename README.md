# Computer Vision: CNN & Vision Transformer (ViT)

This project implements and compares deep learning architectures for image classification, focusing on **Convolutional Neural Networks (CNNs)** and a **Vision Transformer (ViT)** trained from scratch on CIFAR-10.

---

## 📋 Overview

The goal of this project is to study and compare different architectural choices in computer vision:

- **CNNs (Convolutional Neural Networks)**: multiple handcrafted architectures with varying depth, kernel sizes, and capacities
- **Vision Transformer (ViT)**: a transformer-based model adapted for image classification using patch embeddings and self-attention

All models are evaluated on the **CIFAR-10 dataset (60,000 images, 10 classes)**.

---

## 🧠 Models

### CNN architectures

Six CNN variants were implemented to study the impact of architectural design choices:

| Model | Description | Parameters |
|------|------------|-----------|
| CNN1 | Simple baseline CNN | ~14K |
| CNN2 | Reduced channel capacity | ~6K |
| CNN3 | Small filters (3×3) | ~5K |
| CNN4 | Single conv layer + FC | ~41K |
| LeNet | Classical CNN (1998) | ~83K |
| MyCNN | Deep CNN with skip connections and dropout | ~1.16M |

Key observations:
- Increasing depth and channel width significantly increases model capacity
- Fully connected layers dominate parameter count in shallow models
- MyCNN provides the strongest CNN baseline due to higher representational power

---

### Vision Transformer (ViT)

A Vision Transformer was implemented from scratch with the following configuration:

- Patch size: 4
- Image size: 32×32
- Hidden dimension: 128
- Attention heads: 8
- Transformer layers: 12
- Dropout: 0.1

Total parameters: ~2.26M

The ViT uses:
- Patch embedding via convolution
- Learnable positional encoding
- Multi-head self-attention blocks
- A classification token ([CLS])

---

## ⚙️ Training setup

All models were trained under comparable conditions:

- Optimizer: Adam (CNNs), AdamW (ViT)
- Batch size: 64
- Learning rate:
  - CNNs: 1e-3
  - ViT: 3e-4
- Dataset: CIFAR-10
- Loss: CrossEntropyLoss

ViT was trained for more epochs due to slower convergence behavior.

---

## 📊 Results summary

| Model | Behavior | Performance |
|------|---------|-------------|
| CNNs | Fast convergence, stable training | Higher accuracy overall |
| ViT | Slower convergence, more unstable early training | ~74% test accuracy |

---

## 🔍 Key findings

CNNs outperform the Vision Transformer in this setup for several reasons:

### 1. Dataset size limitation
CIFAR-10 is relatively small for transformer-based architectures. ViTs typically require large-scale datasets to learn meaningful attention patterns effectively.

---

### 2. Inductive bias
CNNs benefit from strong built-in assumptions:
- Local spatial structure
- Translation invariance

ViTs lack these priors and must learn them from data, making them less efficient in low-data regimes.

---

### 3. Optimization sensitivity
ViTs are more sensitive to:
- learning rate choice
- number of training epochs
- regularization strategy

In this project, both architectures were trained from scratch under similar constraints, which favored CNNs.

---

### 4. Data efficiency
CNNs are significantly more data-efficient, while ViTs require more data or pretraining to reach optimal performance.

---

## 📈 Conclusion

This project highlights the trade-off between CNNs and Vision Transformers:

- CNNs are efficient, stable, and well-suited for small to medium datasets
- ViTs are more expressive but require larger datasets and more careful training

Although the ViT has a significantly higher number of parameters (~2.26M), it does not outperform CNNs on CIFAR-10 under identical training conditions due to data limitations and optimization constraints.

---

## 🚀 Future work

- Pretraining ViT on larger datasets (e.g. ImageNet)
- Hybrid CNN-Transformer architectures
- Improved regularization techniques for ViT
- Extending experiments to larger datasets (CIFAR-100, ImageNet subsets)
- Adding formal benchmarking (FLOPs, inference time, latency)

---