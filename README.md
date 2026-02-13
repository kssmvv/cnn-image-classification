# cnn-image-classification

CNN and transfer learning experiments in PyTorch — training a small CNN on MNIST from scratch, and fine-tuning MobileNetV2 on Oxford Flowers 102.

## Overview

This project contains two self-contained Jupyter notebooks that explore image classification at different scales:

1. **MNIST from scratch** — build a tiny convolutional neural network (~32K parameters) and study how architectural choices affect performance.
2. **Flowers 102 via transfer learning** — take a MobileNetV2 pretrained on ImageNet and fine-tune it to classify 102 flower species, reaching ~86% validation accuracy.

Both notebooks are designed to run on CPU (MNIST) or CPU/GPU (Flowers) and download their datasets automatically on first run.

## Project Structure

```
.
├── mnist_cnn.ipynb                 # Small CNN on MNIST + ablation experiments
├── flowers_transfer_learning.ipynb # MobileNetV2 transfer learning on Flowers 102
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/cnn-image-classification.git
cd cnn-image-classification

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Datasets (MNIST and Flowers 102) are downloaded automatically into a `data/` directory on first run.

## Notebooks

### 1. `mnist_cnn.ipynb` — Small CNN on MNIST

Builds a minimal two-layer CNN and trains it on a 10K-sample subset of MNIST.

**What's inside:**
- Data loading and visualization
- A `SmallCNN` model with configurable channel counts
- Training loop with Adam optimizer (5 epochs, lr=0.01)
- Test evaluation with confusion matrix
- Four ablation experiments:

| Experiment | Parameters | Best Val Acc | Test Acc |
|---|---|---|---|
| Baseline (c1=8, c2=16, k=3) | 32,618 | 97.4% | 96.7% |
| More channels (c1=16, c2=32) | 67,530 | 96.2% | 94.7% |
| Larger kernel (5x5) | 34,794 | 95.6% | 94.9% |
| Dropout (p=0.2) | 32,618 | 96.7% | 96.1% |
| Reduced data (2K samples) | 32,618 | 94.5% | 93.6% |

**Key takeaway:** The baseline architecture is already well-suited for MNIST. Doubling channels or increasing kernel size adds parameters without improving accuracy. Dropout provides marginal regularization benefit, while reducing training data clearly hurts generalization.

### 2. `flowers_transfer_learning.ipynb` — Transfer Learning on Flowers 102

Fine-tunes a pretrained MobileNetV2 on the Oxford Flowers 102 dataset (102 classes, ~1K training images).

**What's inside:**
- Dataset loading and exploration
- Training augmentations (random crop, horizontal flip, Gaussian blur)
- MobileNetV2 with a replaced classifier head (1280 → 102)
- Training with Adam + StepLR scheduler (10 epochs)
- Test evaluation and sample predictions (correct vs. incorrect)
- Comparison with an aggressive augmentation pipeline (adds vertical flip, rotation, stronger blur)

**Results:**

| Setup | Val Acc | Test Acc |
|---|---|---|
| Standard augmentations (10 epochs) | 86.1% | 83.2% |
| Aggressive augmentations (5 epochs) | 70.5% | 68.3% |

**Key takeaway:** With only ~1K training images, moderate augmentations combined with a strong pretrained backbone work well. Overly aggressive augmentations (rotation, vertical flips for flowers) actually hurt — the model needs more epochs to learn from heavily distorted inputs.

## Tech Stack

- **PyTorch** + **torchvision** for models, datasets, and transforms
- **NumPy** for array operations
- **Matplotlib** for visualization
- **scikit-learn** for confusion matrix
- **pandas** for result tables
