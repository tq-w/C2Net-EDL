# C2Net-EDL: 基于证据深度学习的长尾多标签医学图像分类

## 项目概述

C2Net-EDL 在 C2Net 基础上引入证据深度学习（Evidential Deep Learning, EDL），实现不确定性感知的多标签眼底疾病分类。模型输出 Beta 分布参数 (α, β) 而非点估计，为临床决策支持提供不确定性量化能力。

## 环境配置

```bash
# 创建conda环境
conda create -n c2net_edl python=3.10 -y
conda activate c2net_edl

# 安装PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install einops timm pandas scikit-learn PyYAML tensorboard matplotlib packaging munkres
```

## 数据准备

### ODIR 20类细粒度数据集

主任务使用从 ODIR 数据集派生的 20 个细粒度疾病类别。安全标签（laser_spot, lens_dust, low_image_quality, post_laser_photocoagulation, post_retinal_laser_surgery）从主任务中排除。罕见视网膜变体合并入 other_retinal_diseases。

| 索引 | 类别 | 样本数 | 索引 | 类别 | 样本数 |
|------|------|--------|------|------|--------|
| 0 | normal (正常) | 4374 | 10 | drusen (玻璃疣) | 156 |
| 1 | diabetic_retinopathy (糖尿病视网膜病变) | 2550 | 11 | maculopathy (黄斑病变) | 84 |
| 2 | glaucoma (青光眼) | 547 | 12 | myelinated_nerve_fibers (有髓神经纤维) | 68 |
| 3 | cataract (白内障) | 476 | 13 | optic_disc_edema (视盘水肿) | 49 |
| 4 | age_related_macular_degeneration (年龄相关性黄斑变性) | 455 | 14 | optic_nerve_atrophy (视神经萎缩) | 42 |
| 5 | hypertensive_retinopathy (高血压性视网膜病变) | 454 | 15 | peripapillary_atrophy (视乳头旁萎缩) | 32 |
| 6 | pathological_myopia (病理性近视) | 411 | 16 | pigmentation_disorder (色素沉着异常) | 26 |
| 7 | retinal_occlusion (视网膜血管阻塞) | 375 | 17 | refractive_media_opacity (屈光介质混浊) | 21 |
| 8 | epiretinal_membrane (视网膜前膜) | 277 | 18 | vitreous_degeneration (玻璃体变性) | 266 |
| 9 | chorioretinal_atrophy (脉络膜视网膜萎缩) | 249 | 19 | other_retinal_diseases (其他视网膜疾病) | 44 |

总计：9720 张图像（280 张仅含安全标签的图像已移除）

### MuReD 20类数据集

MuReD (多标签视网膜疾病) 数据集包含 2208 张图像，20 个疾病类别。

| 类别 | 样本数 | 类别 | 样本数 |
|------|--------|------|--------|
| DR (糖尿病视网膜病变) | 495 | CRVO (视网膜中央静脉阻塞) | 55 |
| NORMAL (正常) | 493 | CNV (脉络膜新生血管) | 60 |
| MH (黄斑裂孔) | 169 | RS (视网膜脱离) | 58 |
| ODC (视盘改变) | 263 | ODE (视盘水肿) | 57 |
| TSLN (颞侧视网膜神经纤维层) | 156 | LS (激光瘢痕) | 46 |
| ARMD (年龄相关性黄斑变性) | 158 | CSR (中心性浆液性视网膜病变) | 36 |
| DN (糖尿病视网膜病变) | 162 | HTR (高血压性视网膜病变) | 35 |
| MYA (近视) | 89 | ASR (年龄相关性病变) | 33 |
| BRVO (视网膜分支静脉阻塞) | 79 | CRS (脉络膜视网膜病变) | 30 |
| ODP (视盘异常) | 62 | OTHER (其他) | 261 |

### 数据目录结构

```
data/
├── ODIR_crop/              # ODIR眼底图像 (xxx_left.png, xxx_right.png)
└── MuReD/                  # MuReD眼底图像 (ID.png)
datasets/
├── ODIR_20class.csv        # ODIR 20类标签 (image_name, labels, fold)
├── MuReD_20class.csv       # MuReD 20类标签 (ID, 类别列, fold)
└── label_mapping.csv       # 原始30类映射
```

### CSV格式

ODIR CSV 包含：
- `image_name`: 图像文件名（如 `0_left.png`）
- `labels`: 分号分隔的类别索引（如 `1;7`）
- `fold`: 五折交叉验证的折号（0-4）

```csv
image_name,labels,fold
0_left.jpg,4,2
0_right.jpg,0,2
2_left.jpg,1;7,1
```

### 生成5折交叉验证数据

```bash
# 生成ODIR 5折划分（患者级别）
python scripts/generate_odir_5fold.py \
    --input_csv /path/to/ODIR_20class_server.csv \
    --output_csv datasets/ODIR_20class.csv \
    --n_folds 5 \
    --seed 42

# 生成MuReD 5折划分（患者级别）
python scripts/generate_mured_5fold.py \
    --train_csv /path/to/mured/train_data.csv \
    --val_csv /path/to/mured/val_data.csv \
    --output_csv datasets/MuReD_20class.csv \
    --n_folds 5 \
    --seed 42
```

## 配置说明

编辑 `config.py` 设置数据路径：

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

## 训练

```bash
conda activate c2net_edl

# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python train.py --dataset odir --fold 0 --device cuda

# 后台训练
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset odir --fold 0 --device cuda > train.log 2>&1 &
```

### 监控训练

```bash
# 查看日志
tail -f outputs/c2net_edl/log.txt

# 启动TensorBoard
tensorboard --logdir outputs/c2net_edl/tensorboard --port 6006
```

### 输出文件

```
outputs/c2net_edl/
├── checkpoint-best.pth     # 最佳模型权重
├── checkpoint-latest.pth   # 最新模型权重
├── log.txt                 # 训练日志
└── tensorboard/            # TensorBoard日志
```

## 模型架构

### 双分支Beta证据头

独立的α和β分支，无权重共享：

```
输入 (ViT特征)
  ├── alpha_head (MLP) → z_α → α = σ·Softplus(z_α) + 1
  └── beta_head  (MLP) → z_β → β = σ·Softplus(z_β) + 1
```

- α分支偏置初始化为 -2.0（较弱的初始正向证据）
- β分支偏置初始化为 0（均匀先验）
- 证据缩放因子 σ = 10.0

### 损失函数

```
L = L_risk + λ(t)·L_KL + λ_contra·L_contra
```

- **L_risk**: 非对称证据风险损失 (w_neg = 0.2)
- **L_KL**: KL(Beta(α,β) || Beta(1,1))，包含完整的Digamma项
- **λ(t)**: 周期性KL退火（4个周期，每个周期前半段预热）
- **L_contra**: 监督对比损失（权重 = 0.1）

### 训练细节

| 设置 | 值 |
|------|-----|
| 主干网络 | ViT-Base (ImageNet-21k预训练) |
| 输入尺寸 | 256 × 256 |
| 优化器 | AdamW (lr=1e-4, weight_decay=0.05) |
| 学习率调度 | 10 epoch线性预热 + 余弦退火 |
| 早停 | patience=15 |
| 混合精度 | 启用AMP |
| 归一化 | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |
| 数据增强 | 水平翻转(p=0.5)、旋转(±15°)、色彩抖动(±10%)、高斯噪声(std=0.01) |

## 评估指标

| 指标 | 说明 |
|------|------|
| mAP | 平均精度均值 |
| AR | 平均召回率 |
| AF1 | 平均F1分数 |
| AK | 平均Cohen's Kappa |
| ACC | 准确率 |
| ECE↓ | 期望校准误差（越低越好） |
| BS↓ | Brier分数（越低越好） |

## 推理示例

```python
import torch
from losses.edl_loss import compute_probability_from_evidence, compute_predictive_uncertainty

# 加载模型
checkpoint = torch.load('outputs/c2net_edl/checkpoint-best.pth')
model.load_state_dict(checkpoint['model'])

with torch.no_grad():
    alpha, beta = model(images)             # (B, 20)
    prob = alpha / (alpha + beta)           # 预测概率
    uncertainty = 2.0 / (alpha + beta)      # 不确定性分数
    predictions = (prob >= 0.5).float()     # 二值预测
```

## 五折交叉验证

### 运行5折交叉验证

```bash
# ODIR 5折交叉验证
python scripts/run_5fold_cv.py \
    --dataset odir \
    --config configs/full_edl.yaml \
    --output_dir results/

# MuReD 5折交叉验证
python scripts/run_5fold_cv.py \
    --dataset mured \
    --config configs/full_edl.yaml \
    --output_dir results/
```

## 消融实验

### 配置文件

消融实验配置存储在 `configs/ablation/` 目录：

| 配置文件 | 损失类型 | neg_weight | evidence_scale | annealing_type | contra_weight |
|----------|----------|------------|----------------|----------------|---------------|
| sigmoid_baseline.yaml | bce | - | - | - | 0 |
| evidence_modeling.yaml | edl_beta | 1.0 | 1.0 | linear | 0 |
| asymmetric_weighting.yaml | edl_beta | 0.2 | 1.0 | linear | 0 |
| evidence_scaling.yaml | edl_beta | 0.2 | 10.0 | linear | 0 |
| cyclical_annealing.yaml | edl_beta | 0.2 | 10.0 | cyclical | 0 |
| full_edl.yaml | edl_beta | 0.2 | 10.0 | cyclical | 0.1 |

### 运行消融实验

```bash
# 运行所有消融配置
python scripts/run_ablation.py \
    --dataset odir \
    --config_dir configs/ablation/ \
    --output_dir results/
```

## 评估脚本

### 超参数敏感性分析

```bash
python scripts/run_hyperparameter_analysis.py \
    --dataset odir \
    --output_dir results/
```

### OOD检测评估

```bash
python scripts/run_ood_detection.py \
    --checkpoint outputs/c2net_edl/checkpoint-best.pth \
    --output_dir results/
```

### 临床分层评估

```bash
python scripts/run_clinical_triage.py \
    --checkpoint outputs/c2net_edl/checkpoint-best.pth \
    --output_dir results/
```

## 实验结果

实验结果存储在 `results/` 目录：

| 文件 | 说明 |
|------|------|
| odir_main_results.csv | ODIR主实验结果（表4.3） |
| mured_main_results.csv | MuReD主实验结果（表4.4） |
| ablation_results.csv | 消融实验结果（表4.5） |
| per_class_performance.csv | 每类性能对比（表4.6） |
| ood_detection_results.csv | OOD检测结果（表4.7） |
| clinical_triage_results.csv | 临床分层结果（表4.8） |
| hyperparameter_sensitivity.csv | 超参数敏感性分析（图4.7） |
| fold_results_odir.csv | ODIR五折交叉验证结果 |
| fold_results_mured.csv | MuReD五折交叉验证结果 |

## 项目结构

```
C2Net_EDL/
├── configs/
│   └── ablation/              # 消融实验配置文件
├── data/
│   ├── ODIR_crop/             # ODIR眼底图像
│   └── MuReD/                 # MuReD眼底图像
├── datasets/
│   ├── ODIR_20class.csv       # ODIR 20类标签（含fold列）
│   ├── MuReD_20class.csv      # MuReD 20类标签（含fold列）
│   └── label_mapping.csv      # 原始30类映射
├── losses/
│   └── edl_loss.py            # EDL损失函数
├── models/
│   ├── head.py                # EDL头（双分支Beta证据）
│   └── vision_transformer.py  # ViT主干网络
├── scripts/
│   ├── generate_odir_5fold.py     # 生成ODIR五折划分
│   ├── generate_mured_5fold.py    # 生成MuReD五折划分
│   ├── run_ablation.py            # 运行消融实验
│   ├── run_5fold_cv.py            # 运行五折交叉验证
│   ├── run_hyperparameter_analysis.py  # 超参数敏感性分析
│   ├── run_ood_detection.py       # OOD检测评估
│   └── run_clinical_triage.py     # 临床分层评估
├── utils/
│   ├── metrics.py             # 评估指标
│   └── ...
├── results/                   # 实验结果
├── config.py                  # 配置文件
├── train.py                   # 主训练脚本
└── README.md
```

## 服务器部署

### 服务器环境配置

```bash
# SSH连接服务器
ssh user@server

# 克隆仓库
git clone https://github.com/tq-w/C2Net-EDL.git
cd C2Net-EDL

# 创建conda环境
conda create -n c2net_edl python=3.10 -y
conda activate c2net_edl

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops timm pandas scikit-learn PyYAML tensorboard matplotlib packaging munkres

# 修改config.py中的数据路径
# 将 '/root/C2Net_EDL/data/ODIR_crop' 改为实际路径
```

### GPU训练命令

```bash
# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python train.py --dataset odir --fold 0 --device cuda

# 多GPU训练（2块GPU）
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset odir --fold 0 --distributed --device cuda

# 后台训练
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset odir --fold 0 --device cuda > train.log 2>&1 &

# 监控训练
tail -f outputs/c2net_edl/log.txt
tensorboard --logdir outputs/c2net_edl/tensorboard --port 6006
```

### 运行完整实验

```bash
# 运行5折交叉验证
for fold in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset odir --fold $fold --device cuda --task odir_fold_$fold
done

# 运行消融实验
python scripts/run_ablation.py --dataset odir --config_dir configs/ablation/

# 运行超参数分析
python scripts/run_hyperparameter_analysis.py --dataset odir
```

## 常见问题

### Q: 如何使用自己的数据集？

1. 准备图像放在 `data/your_dataset/` 目录
2. 创建CSV文件，包含列：`image_name`, `labels`（分号分隔的索引）
3. 添加 `fold` 列用于五折交叉验证
4. 更新 `config.py` 中的数据路径

### Q: 如何修改类别数？

1. 更新 `config.py` 中的 `num_classes`
2. 更新损失配置（alpha, gamma, pos_weight数组）
3. 更新数据集类中的类别名称

### Q: 训练损失为负数，这正常吗？

是的！EDL损失可以为负数，因为它使用Digamma函数。损失应该随时间下降。

### Q: 如何解读不确定性？

- **低不确定性 (< 0.3)**：模型置信度高，预测可靠
- **中等不确定性 (0.3-0.6)**：模型置信度较低，建议人工复核
- **高不确定性 (> 0.6)**：模型不确定，建议专家会诊

### Q: GPU内存不足？

- 减小 `config.py` 中的 `batch_size`
- 使用 `--accum_iter 2` 进行梯度累积
- 使用混合精度训练（默认启用）

## 参考文献

- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
- C2Net: Constraint-Aware Contrastive Learning for Multi-Label Medical Image Classification

## 引用

```bibtex
@article{c2net_edl,
  title={C2Net-EDL: Evidential Deep Learning for Long-Tail Multi-Label Medical Image Classification},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```
