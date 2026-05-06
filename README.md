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

### Dataset: ODIR 20-Class Fine-Grained

The main task uses 20 fine-grained disease classes derived from the ODIR dataset. Safety labels (laser_spot, lens_dust, low_image_quality, post_laser_photocoagulation, post_retinal_laser_surgery) are excluded from the main task. Rare retinal variants (retinal_pigmentation, retinitis_pigmentosa, spotted_membranous_change, tessellated_fundus, white_vessel) are merged into other_retinal_diseases.

| Index | Class | Samples | Index | Class | Samples |
|-------|-------|---------|-------|-------|---------|
| 0 | normal | 4374 | 10 | drusen | 156 |
| 1 | diabetic_retinopathy | 2550 | 11 | maculopathy | 84 |
| 2 | glaucoma | 547 | 12 | myelinated_nerve_fibers | 68 |
| 3 | cataract | 476 | 13 | optic_disc_edema | 49 |
| 4 | age_related_macular_degeneration | 455 | 14 | optic_nerve_atrophy | 42 |
| 5 | hypertensive_retinopathy | 454 | 15 | peripapillary_atrophy | 32 |
| 6 | pathological_myopia | 411 | 16 | pigmentation_disorder | 26 |
| 7 | retinal_occlusion | 375 | 17 | refractive_media_opacity | 21 |
| 8 | epiretinal_membrane | 277 | 18 | vitreous_degeneration | 266 |
| 9 | chorioretinal_atrophy | 249 | 19 | other_retinal_diseases | 44 |

Total: 9720 images (280 safety-label-only images removed)

### Data Structure

```
data/
└── ODIR_crop/              # Fundus images (xxx_left.png, xxx_right.png)
datasets/
├── ODIR_20class.csv        # 20-class labels (image_name, labels)
└── label_mapping.csv       # 30-class original mapping
```

### CSV Format

The CSV contains:
- `image_name`: image filename (e.g., `0_left.png`)
- `labels`: semicolon-separated class indices (e.g., `1;7`)

```csv
image_name,labels
0_left.jpg,4
0_right.jpg,0
2_left.jpg,1;7
```

Note: `random_mode` column is optional. If absent, the code auto-splits 80/10/10.

### Generate 20-Class CSV from 30-Class Labels

```python
import pandas as pd
from collections import Counter

df = pd.read_csv('datasets/train_labels_multilabel.csv')  # 30-class original

safety_labels = {11, 12, 13, 20, 21}
merge_to_other = {23, 24, 25, 26, 28}

old_to_new = {
    0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10,
    14:11, 15:12, 16:13, 17:14, 18:15, 19:16, 22:17, 27:18, 29:19
}
for idx in merge_to_other:
    old_to_new[idx] = 19

new_rows = []
for _, row in df.iterrows():
    old_labels = [int(x) for x in str(row['labels']).split(';') if x.strip()]
    non_safety = [l for l in old_labels if l not in safety_labels]
    if not non_safety:
        continue
    new_labels = sorted(set(old_to_new.get(l, 19) for l in non_safety))
    new_rows.append({
        'image_name': row['image_name'],
        'labels': ';'.join(str(l) for l in new_labels)
    })

pd.DataFrame(new_rows).to_csv('datasets/ODIR_20class.csv', index=False)
```

## Configuration

Edit `config.py` to set data paths:

```python
self.data = {
    'root_dir': '/path/to/ODIR_crop',
    'csv_file': 'datasets/ODIR_20class.csv',
    'normalization': 'imagenet',
}

self.model = {
    'type': 'vit_base',
    'num_classes': 20,
    'input_size': 256,
}

self.training = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-4,
    'warmup_epochs': 10,
    'early_stop_patience': 15,
}
```

## Training

```bash
conda activate c2net_edl

# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py --device cuda

# Background training
CUDA_VISIBLE_DEVICES=0 nohup python train.py --device cuda > train.log 2>&1 &
```

### Monitor

```bash
tail -f outputs/c2net_edl/log.txt
tensorboard --logdir outputs/c2net_edl/tensorboard --port 6006
```

### Outputs

```
outputs/c2net_edl/
├── checkpoint-best.pth
├── checkpoint-latest.pth
├── log.txt
└── tensorboard/
```

## Architecture

### Dual-Branch Beta Evidence Head

Independent alpha and beta branches with no weight sharing:

```
Input (ViT features)
  ├── alpha_head (MLP) → z_α → α = σ·Softplus(z_α) + 1
  └── beta_head  (MLP) → z_β → β = σ·Softplus(z_β) + 1
```

- α bias initialized to -2.0 (weaker initial positive evidence)
- β bias initialized to 0 (uniform prior)
- Evidence scale σ = 10.0

### Loss Function

```
L = L_risk + λ(t)·L_KL + λ_contra·L_contra
```

- **L_risk**: Asymmetric evidence risk loss (w_neg = 0.2)
- **L_KL**: KL(Beta(α,β) || Beta(1,1)) with full Digamma terms
- **λ(t)**: Cyclical KL annealing (4 cycles, warmup for first half)
- **L_contra**: Supervised contrastive loss (weight = 0.1)

### Training Details

| Setting | Value |
|---------|-------|
| Backbone | ViT-Base (ImageNet-21k pretrained) |
| Input size | 256 x 256 |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.05) |
| LR schedule | 10-epoch linear warmup + cosine annealing |
| Early stopping | patience=15 |
| Mixed precision | AMP enabled |
| Normalization | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |
| Augmentation | H-flip (p=0.5), rotation (±15°), color jitter (±10%), Gaussian noise (std=0.01) |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| F1 Score | Macro-averaged F1 |
| ROC AUC | Area under ROC curve |
| ECE | Expected Calibration Error |
| Brier Score | Probability prediction quality |

## Inference

```python
import torch
from losses.edl_loss import compute_probability_from_evidence, compute_predictive_uncertainty

checkpoint = torch.load('outputs/c2net_edl/checkpoint-best.pth')
model.load_state_dict(checkpoint['model'])

with torch.no_grad():
    alpha, beta = model(images)             # (B, 20)
    prob = alpha / (alpha + beta)           # predicted probability
    uncertainty = 2.0 / (alpha + beta)      # uncertainty score
    predictions = (prob >= 0.5).float()
```

## References

- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
- C2Net: Constraint-Aware Contrastive Learning for Multi-Label Medical Image Classification
