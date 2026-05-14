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

## 5-Fold Cross-Validation

### Generate 5-Fold Splits

```bash
# Generate ODIR 5-fold splits (patient-level)
python scripts/generate_odir_5fold.py \
    --input_csv /tmp/ODIR_20class_server.csv \
    --output_csv datasets/ODIR_20class.csv \
    --n_folds 5 \
    --seed 42

# Generate MuReD 5-fold splits (patient-level)
python scripts/generate_mured_5fold.py \
    --train_csv /path/to/DataProcess/mured/train_data.csv \
    --val_csv /path/to/DataProcess/mured/val_data.csv \
    --output_csv datasets/MuReD_20class.csv \
    --n_folds 5 \
    --seed 42
```

### Run 5-Fold Cross-Validation

```bash
# Run 5-fold CV for ODIR
python scripts/run_5fold_cv.py \
    --dataset odir \
    --config configs/full_edl.yaml \
    --output_dir results/

# Run 5-fold CV for MuReD
python scripts/run_5fold_cv.py \
    --dataset mured \
    --config configs/full_edl.yaml \
    --output_dir results/
```

## Ablation Studies

### Configuration Files

Ablation configurations are stored in `configs/ablation/`:

| Configuration | Loss Type | neg_weight | evidence_scale | annealing_type | contra_weight |
|---------------|-----------|------------|----------------|----------------|---------------|
| sigmoid_baseline.yaml | bce | - | - | - | 0 |
| evidence_modeling.yaml | edl_beta | 1.0 | 1.0 | linear | 0 |
| asymmetric_weighting.yaml | edl_beta | 0.2 | 1.0 | linear | 0 |
| evidence_scaling.yaml | edl_beta | 0.2 | 10.0 | linear | 0 |
| cyclical_annealing.yaml | edl_beta | 0.2 | 10.0 | cyclical | 0 |
| full_edl.yaml | edl_beta | 0.2 | 10.0 | cyclical | 0.1 |

### Run Ablation Studies

```bash
# Run all ablation configurations
python scripts/run_ablation.py \
    --dataset odir \
    --config_dir configs/ablation/ \
    --output_dir results/
```

## Evaluation Scripts

### Hyperparameter Sensitivity Analysis

```bash
python scripts/run_hyperparameter_analysis.py \
    --dataset odir \
    --output_dir results/
```

### OOD Detection Evaluation

```bash
python scripts/run_ood_detection.py \
    --checkpoint outputs/c2net_edl/checkpoint-best.pth \
    --output_dir results/
```

### Clinical Triage Evaluation

```bash
python scripts/run_clinical_triage.py \
    --checkpoint outputs/c2net_edl/checkpoint-best.pth \
    --output_dir results/
```

## Experiment Results

Experiment results are stored in `results/`:

| File | Description |
|------|-------------|
| odir_main_results.csv | ODIR main experiment results (Table 4.3) |
| mured_main_results.csv | MuReD main experiment results (Table 4.4) |
| ablation_results.csv | Ablation study results (Table 4.5) |
| per_class_performance.csv | Per-class performance comparison (Table 4.6) |
| ood_detection_results.csv | OOD detection results (Table 4.7) |
| clinical_triage_results.csv | Clinical triage results (Table 4.8) |
| hyperparameter_sensitivity.csv | Hyperparameter sensitivity analysis (Figure 4.7) |
| fold_results_odir.csv | ODIR 5-fold CV results |
| fold_results_mured.csv | MuReD 5-fold CV results |

## Project Structure

```
C2Net_EDL/
├── configs/
│   └── ablation/              # Ablation study configurations
│       ├── sigmoid_baseline.yaml
│       ├── evidence_modeling.yaml
│       ├── asymmetric_weighting.yaml
│       ├── evidence_scaling.yaml
│       ├── cyclical_annealing.yaml
│       └── full_edl.yaml
├── data/
│   ├── ODIR_crop/             # ODIR fundus images
│   └── MuReD/                 # MuReD fundus images
├── datasets/
│   ├── ODIR_20class.csv       # ODIR 20-class labels with fold column
│   ├── MuReD_20class.csv      # MuReD 20-class labels with fold column
│   └── label_mapping.csv      # Original 30-class mapping
├── losses/
│   └── edl_loss.py            # EDL loss functions
├── models/
│   ├── head.py                # EDL head (dual-branch Beta evidence)
│   └── vision_transformer.py  # ViT backbone
├── scripts/
│   ├── generate_odir_5fold.py     # Generate ODIR 5-fold splits
│   ├── generate_mured_5fold.py    # Generate MuReD 5-fold splits
│   ├── run_ablation.py            # Run ablation studies
│   ├── run_5fold_cv.py            # Run 5-fold cross-validation
│   ├── run_hyperparameter_analysis.py  # Hyperparameter sensitivity
│   ├── run_ood_detection.py       # OOD detection evaluation
│   └── run_clinical_triage.py     # Clinical triage evaluation
├── utils/
│   ├── metrics.py             # Evaluation metrics
│   └── ...
├── results/                   # Experiment results
├── config.py                  # Configuration
├── train.py                   # Main training script
└── README.md
```

## MuReD Dataset

MuReD (Multi-label Retinal Disease) dataset contains 2208 images with 20 disease classes.

| Class | Samples | Class | Samples |
|-------|---------|-------|---------|
| DR | 495 | CRVO | 55 |
| NORMAL | 493 | CNV | 60 |
| MH | 169 | RS | 58 |
| ODC | 263 | ODE | 57 |
| TSLN | 156 | LS | 46 |
| ARMD | 158 | CSR | 36 |
| DN | 162 | HTR | 35 |
| MYA | 89 | ASR | 33 |
| BRVO | 79 | CRS | 30 |
| ODP | 62 | OTHER | 261 |

### MuReD Data Preparation

```bash
# MuReD data should be in data/MuReD/ directory
# Images are named as: {ID}.png (e.g., aria_c_25_1.png)

# Generate 5-fold splits
python scripts/generate_mured_5fold.py \
    --train_csv /path/to/mured/train_data.csv \
    --val_csv /path/to/mured/val_data.csv \
    --output_csv datasets/MuReD_20class.csv
```

## Server Deployment

### Environment Setup on Server

```bash
# SSH to server
ssh user@server

# Clone repository
git clone https://github.com/tq-w/C2Net-EDL.git
cd C2Net-EDL

# Create conda environment
conda create -n c2net_edl python=3.10 -y
conda activate c2net_edl

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops timm pandas scikit-learn PyYAML tensorboard matplotlib packaging munkres

# Update data path in config.py
# Change '/root/C2Net_EDL/data/ODIR_crop' to your actual path
```

### Training on Server with GPU

```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python train.py --dataset odir --fold 0 --device cuda

# Multi-GPU training (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset odir --fold 0 --distributed --device cuda

# Background training
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset odir --fold 0 --device cuda > train.log 2>&1 &

# Monitor training
tail -f outputs/c2net_edl/log.txt
tensorboard --logdir outputs/c2net_edl/tensorboard --port 6006
```

### Run Full Experiments

```bash
# Run 5-fold cross-validation
for fold in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset odir --fold $fold --device cuda --task odir_fold_$fold
done

# Run ablation studies
python scripts/run_ablation.py --dataset odir --config_dir configs/ablation/

# Run hyperparameter analysis
python scripts/run_hyperparameter_analysis.py --dataset odir
```

## FAQ

### Q: How to use my own dataset?

1. Prepare images in `data/your_dataset/` directory
2. Create CSV file with columns: `image_name`, `labels` (semicolon-separated indices)
3. Add `fold` column for 5-fold cross-validation
4. Update `config.py` with your data paths

### Q: How to change number of classes?

1. Update `num_classes` in `config.py`
2. Update loss configuration (alpha, gamma, pos_weight arrays)
3. Update class names in dataset class

### Q: Training loss is negative, is this normal?

Yes! EDL loss can be negative because it uses Digamma functions. The loss should decrease over time.

### Q: How to interpret uncertainty?

- **Low uncertainty (< 0.3)**: Model is confident, prediction is reliable
- **Medium uncertainty (0.3-0.6)**: Model is less confident, consider manual review
- **High uncertainty (> 0.6)**: Model is uncertain, recommend specialist consultation

### Q: GPU memory error?

- Reduce `batch_size` in config
- Use `--accum_iter 2` for gradient accumulation
- Use mixed precision training (enabled by default)

## References

- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
- C2Net: Constraint-Aware Contrastive Learning for Multi-Label Medical Image Classification

## Citation

```bibtex
@article{c2net_edl,
  title={C2Net-EDL: Evidential Deep Learning for Long-Tail Multi-Label Medical Image Classification},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```
