#!/bin/bash
# C2Net-EDL 演示脚本
# 展示所有可以运行的内容

echo "=========================================="
echo "C2Net-EDL 代码审查演示"
echo "=========================================="

echo ""
echo "1. 项目结构:"
echo "------------------------------------------"
tree -L 2 -d

echo ""
echo "2. 配置文件测试:"
echo "------------------------------------------"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate py312

python3 -c "
import yaml
configs = [
    'configs/ablation/sigmoid_baseline.yaml',
    'configs/ablation/evidence_modeling.yaml',
    'configs/ablation/asymmetric_weighting.yaml',
    'configs/ablation/evidence_scaling.yaml',
    'configs/ablation/cyclical_annealing.yaml',
    'configs/ablation/full_edl.yaml'
]

for config_file in configs:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print(f'✅ {config_file}: loss_type={config[\"loss\"][\"loss_type\"]}')
"

echo ""
echo "3. 脚本帮助信息:"
echo "------------------------------------------"
echo "run_ablation.py:"
python scripts/run_ablation.py --help | head -10

echo ""
echo "run_5fold_cv.py:"
python scripts/run_5fold_cv.py --help | head -10

echo ""
echo "4. 数据集信息:"
echo "------------------------------------------"
python3 -c "
import pandas as pd

odir_df = pd.read_csv('datasets/ODIR_20class.csv')
print(f'ODIR数据集: {len(odir_df)} 行')
print(f'Fold分布: {odir_df[\"fold\"].value_counts().sort_index().to_dict()}')

mured_df = pd.read_csv('datasets/MuReD_20class.csv')
print(f'\\nMuReD数据集: {len(mured_df)} 行')
print(f'Fold分布: {mured_df[\"fold\"].value_counts().sort_index().to_dict()}')
"

echo ""
echo "5. 实验结果:"
echo "------------------------------------------"
echo "主实验结果 (ODIR):"
cat results/odir_main_results.csv

echo ""
echo "消融实验结果:"
cat results/ablation_results.csv

echo ""
echo "6. GitHub仓库:"
echo "------------------------------------------"
echo "https://github.com/tq-w/C2Net-EDL"

echo ""
echo "=========================================="
echo "演示完成"
echo "=========================================="
