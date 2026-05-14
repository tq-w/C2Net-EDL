#!/usr/bin/env python3
"""
Run 5-fold cross-validation for C2Net-EDL.

This script runs 5-fold cross-validation and collects results.

Usage:
    python scripts/run_5fold_cv.py \
        --dataset odir \
        --config configs/full_edl.yaml \
        --output_dir results/
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def run_single_fold(dataset, config_path, fold, output_dir):
    """
    Run training for a single fold.
    
    Args:
        dataset: Dataset name (odir or mured)
        config_path: Path to config YAML file
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
    print(f"Running Fold {fold}")
    print(f"{'='*60}")
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Error running fold {fold}:")
            print(result.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout running fold {fold}")
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
                'fold': fold,
                'best_score': metrics.get('Best Score', 0.0),
                'train_loss': metrics.get('Train Loss', 0.0),
                'val_loss': metrics.get('Val Loss', 0.0)
            }
    
    return None


def run_5fold_cv(dataset, config_path, output_dir, n_folds=5):
    """
    Run 5-fold cross-validation.
    
    Args:
        dataset: Dataset name (odir or mured)
        config_path: Path to config YAML file
        output_dir: Output directory for results
        n_folds: Number of folds
    """
    config_name = Path(config_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running 5-fold cross-validation for {dataset}")
    print(f"Config: {config_name}")
    print(f"Output directory: {output_dir}")
    
    # Run all folds
    fold_results = []
    for fold in range(n_folds):
        result = run_single_fold(dataset, config_path, fold, output_dir)
        if result:
            fold_results.append(result)
    
    # Save results
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        
        # Calculate statistics
        best_scores = results_df['best_score'].values
        mean_score = np.mean(best_scores)
        std_score = np.std(best_scores)
        
        # Add mean and std rows
        mean_row = pd.DataFrame([{
            'fold': 'mean',
            'best_score': mean_score,
            'train_loss': results_df['train_loss'].mean(),
            'val_loss': results_df['val_loss'].mean()
        }])
        
        std_row = pd.DataFrame([{
            'fold': 'std',
            'best_score': std_score,
            'train_loss': results_df['train_loss'].std(),
            'val_loss': results_df['val_loss'].std()
        }])
        
        results_df = pd.concat([results_df, mean_row, std_row], ignore_index=True)
        
        # Save to file
        results_file = output_dir / f"fold_results_{dataset}_{config_name}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("5-Fold Cross-Validation Summary")
        print("="*60)
        print(f"\nDataset: {dataset}")
        print(f"Config: {config_name}")
        print(f"\nBest Score: {mean_score:.4f} ± {std_score:.4f}")
        print("\nPer-fold results:")
        for _, row in results_df.iterrows():
            if row['fold'] not in ['mean', 'std']:
                print(f"  Fold {row['fold']}: {row['best_score']:.4f}")
    
    return fold_results


def main():
    parser = argparse.ArgumentParser(description='Run 5-fold cross-validation')
    parser.add_argument('--dataset', type=str, required=True, choices=['odir', 'mured'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds')
    
    args = parser.parse_args()
    
    # Run 5-fold CV
    run_5fold_cv(args.dataset, args.config, args.output_dir, args.n_folds)


if __name__ == '__main__':
    main()
