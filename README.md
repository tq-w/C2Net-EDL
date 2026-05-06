# C2Net-EDL: Evidential Deep Learning for Long-Tail Multi-Label Medical Image Classification

## Overview

C2Net-EDL extends C2Net with Evidential Deep Learning (EDL) for uncertainty-aware multi-label eye disease classification. The model outputs Beta distribution parameters (alpha, beta) instead of point estimates, enabling uncertainty quantification for clinical decision support.

## Environment Setup

```bash
conda create -n c2net_edl python=3.10 -y
conda activate c2net_edl

# PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install einops timm pandas scikit-learn PyYAML tensorboard matplotlib packaging munkres
```

## Data Preparation

### Data Structure

```
data/
└── ODIR_crop/          # Fundus images (xxx_left.png, xxx_right.png)
datasets/
└── ODIR_labels.csv     # CSV with columns: image_name, labels, random_mode, ...
```

### CSV Format

The CSV must contain:
- `image_name`: image filename (e.g., `0_left.png`)
- `labels`: semicolon-separated class indices (e.g., `0;3;7`)
- `random_mode`: data split (`train` / `val` / `test`)

Example:
```csv
image_name,labels,random_mode
0_left.jpg,3,test
0_right.jpg,0,train
1_left.jpg,0,train
1_right.jpg,1;7,val
```

### Convert Existing ODIR CSV

If you have the original ODIR CSV with N/D/G/C/A/H/M/O columns, convert it:

```python
import pandas as pd

df = pd.read_csv('datasets/ODIR_single_v5.csv')
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

def make_labels(row):
    indices = [str(i) for i, c in enumerate(disease_cols) if row[c] == 1]
    return ';'.join(indices) if indices else ''

df['labels'] = df.apply(make_labels, axis=1)
df.to_csv('datasets/ODIR_labels.csv', index=False)
```

## Configuration

Edit `config.py` to set data paths and hyperparameters:

```python
# Data paths
self.data = {
    'root_dir': '/path/to/ODIR_crop',
    'csv_file': '/path/to/ODIR_labels.csv',
    'grade_csv': '/path/to/ODIR_eyeQ_new.csv',  # optional
    'normalization': 'imagenet',
}

# Model
self.model = {
    'type': 'vit_base',
    'num_classes': 8,       # number of disease classes
    'input_size': 256,
    'finetune': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
}

# Training
self.training = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-4,
    'warmup_epochs': 10,
    'early_stop_patience': 15,
}

# EDL
self.edl = {
    'evidence_scale': 10.0,
    'neg_weight': 0.2,
    'contra_weight': 0.1,
    'annealing_type': 'cyclical',
}
```

## Training

```bash
conda activate c2net_edl

# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py --device cuda

#指定 GPU
CUDA_VISIBLE_DEVICES=12 python train.py --device cuda

# Background training with log
CUDA_VISIBLE_DEVICES=12 nohup python train.py --device cuda > train.log 2>&1 &
```

### Training Outputs

Results are saved to `outputs/c2net_edl/`:

```
outputs/c2net_edl/
├── checkpoint-best.pth     # Best model weights
├── checkpoint-latest.pth   # Latest model weights
├── log.txt                 # Training log
└── tensorboard/            # TensorBoard logs
```

### Monitor Training

```bash
# View training log
tail -f outputs/c2net_edl/log.txt

# TensorBoard
tensorboard --logdir outputs/c2net_edl/tensorboard --port 6006
```

### Key Training Details

| Setting | Value |
|---------|-------|
| Backbone | ViT-Base (ImageNet-21k pretrained) |
| Input size | 256 x 256 |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.05) |
| LR schedule | 10-epoch linear warmup + cosine annealing |
| Early stopping | patience=15, based on validation score |
| Mixed precision | Enabled (AMP) |
| Evidence scale | sigma=10.0 |
| Asymmetric weight | w_neg=0.2 |
| KL annealing | Cyclical (4 cycles) |

## Evaluation

Evaluation runs automatically after each epoch during training. Metrics include:

| Metric | Description |
|--------|-------------|
| F1 Score | Macro-averaged F1 |
| ROC AUC | Area under ROC curve |
| ECE | Expected Calibration Error (lower is better) |
| Brier Score | Probability prediction quality (lower is better) |

### Load Trained Model for Inference

```python
import torch
from models.head import EDLHead
from losses.edl_loss import compute_probability_from_evidence, compute_predictive_uncertainty

# Load checkpoint
checkpoint = torch.load('outputs/c2net_edl/checkpoint-best.pth')
model.load_state_dict(checkpoint['model'])

# Forward pass
with torch.no_grad():
    alpha, beta = model(images)  # (B, num_classes)

# Predicted probability and uncertainty
prob = compute_probability_from_evidence(alpha, beta)    # p = alpha / (alpha + beta)
uncertainty = compute_predictive_uncertainty(alpha, beta) # u = 2 / (alpha + beta)

# Binary predictions
predictions = (prob >= 0.5).float()
```

### Uncertainty-Based Triage

```python
# Three-level triage based on uncertainty
low_unc = uncertainty < 0.3       # High confidence -> auto-read
mid_unc = (uncertainty >= 0.3) & (uncertainty < 0.6)  # AI-assisted
high_unc = uncertainty >= 0.6     # Manual review
```

## Hyperparameter Reference

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Evidence scale | `edl.evidence_scale` | 10.0 | Scales evidence dynamic range |
| Negative weight | `edl.neg_weight` | 0.2 | Down-weights negative samples in loss |
| Contrastive weight | `edl.contra_weight` | 0.1 | Weight for supervised contrastive loss |
| KL annealing | `edl.annealing_type` | cyclical | `linear` or `cyclical` |
| KL annealing epochs | `edl.kl_annealing_epochs` | 20 | Reference period for annealing |
| Early stop patience | `training.early_stop_patience` | 15 | Epochs without improvement before stopping |

## References

- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
- C2Net: Constraint-Aware Contrastive Learning for Multi-Label Medical Image Classification
