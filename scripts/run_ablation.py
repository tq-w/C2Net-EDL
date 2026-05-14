#!/usr/bin/env python3
"""
Run ablation studies for C2Net-EDL.

This script runs all ablation configurations and collects results.

Usage:
    python scripts/run_ablation.py \
        --dataset odir \
        --config_dir configs/ablation/ \
        --output_dir results/
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_single_config(config_path, dataset, fold, output_dir):
    """
    Run training for a single configuration and fold.
    
    Args:
        config_path: Path to config YAML file
        dataset: Dataset name (odir or mured)
        fold: Fold number (0-4)
        output_dir: Output directory for results
    
    Returns:
        dict: Results dictionary
    """
    config_name = Path(config_path).stem
    task_name = f"{dataset}_{config_name}_fold{fold}"
    
    # Build command
    cmd = [
        sys.executable, "train.py",
        "--config", str(config_path),
        "--dataset", dataset,
        "--fold", str(fold),
        "--task", task_name,
        "--output_dir", str(output_dir)
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {config_name} - Fold {fold}")
    print(f"{'='*60}")
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Error running {config_name} fold {fold}:")
            print(result.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout running {config_name} fold {fold}")
        return None
    
    # Parse results from log file
    log_file = Path(output_dir) / task_name / "log.txt"
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return None
    
    # Read last line of log for final results
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1].strip()
            # Parse metrics from log line
            # Format: Epoch [X/100] | Train Loss: X.XXXX | Val Loss: X.XXXX | Val Score: X.XXXX | Best Score: X.XXXX
            parts = last_line.split('|')
            metrics = {}
            for part in parts:
                if ':' in part:
                    key, value = part.split(':')
                    key = key.strip()
                    value = value.strip()
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        pass
            
            return {
                'config': config_name,
                'fold': fold,
                'best_score': metrics.get('Best Score', 0.0),
                'train_loss': metrics.get('Train Loss', 0.0),
                'val_loss': metrics.get('Val Loss', 0.0)
            }
    
    return None


def run_ablation_study(dataset, config_dir, output_dir, n_folds=5):
    """
    Run complete ablation study.
    
    Args:
        dataset: Dataset name (odir or mured)
        config_dir: Directory containing config files
        output_dir: Output directory for results
        n_folds: Number of folds
    """
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all config files
    config_files = sorted(config_dir.glob("*.yaml"))
    print(f"Found {len(config_files)} configuration files:")
    for config_file in config_files:
        print(f"  - {config_file.name}")
    
    # Run all configurations
    all_results = []
    for config_file in config_files:
        for fold in range(n_folds):
            result = run_single_config(config_file, dataset, fold, output_dir)
            if result:
                all_results.append(result)
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Round to 3 decimal places
        results_df['best_score'] = results_df['best_score'].round(3)
        results_df['train_loss'] = results_df['train_loss'].round(3)
        results_df['val_loss'] = results_df['val_loss'].round(3)
        
        results_file = output_dir / f"ablation_results_{dataset}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("Ablation Study Summary")
        print("="*60)
        
        # Group by config and calculate mean
        summary = results_df.groupby('config')['best_score'].agg(['mean', 'std']).reset_index()
        summary = summary.sort_values('mean', ascending=False)
        
        print("\nConfig - Mean ± Std")
        print("-" * 40)
        for _, row in summary.iterrows():
            print(f"{row['config']}: {row['mean']:.3f} ± {row['std']:.3f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--dataset', type=str, required=True, choices=['odir', 'mured'],
                        help='Dataset name')
    parser.add_argument('--config_dir', type=str, default='configs/ablation/',
                        help='Directory containing config files')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds')
    
    args = parser.parse_args()
    
    # Run ablation study
    run_ablation_study(args.dataset, args.config_dir, args.output_dir, args.n_folds)


if __name__ == '__main__':
    main()
