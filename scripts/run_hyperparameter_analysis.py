#!/usr/bin/env python3
"""
Run hyperparameter sensitivity analysis for C2Net-EDL.

This script analyzes the sensitivity of key hyperparameters.

Usage:
    python scripts/run_hyperparameter_analysis.py \
        --dataset odir \
        --output_dir results/
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def run_single_hyperparameter(dataset, param_name, param_value, output_dir):
    """
    Run training with a specific hyperparameter value.
    
    Args:
        dataset: Dataset name (odir or mured)
        param_name: Name of hyperparameter
        param_value: Value of hyperparameter
        output_dir: Output directory for results
    
    Returns:
        dict: Results dictionary
    """
    task_name = f"{dataset}_hp_{param_name}_{param_value}"
    
    # Build command based on parameter
    cmd = [
        sys.executable, "train.py",
        "--dataset", dataset,
        "--fold", "0",  # Use fold 0 for speed
        "--task", task_name,
        "--output_dir", str(output_dir)
    ]
    
    # Add hyperparameter-specific arguments
    if param_name == "evidence_scale":
        cmd.extend(["--evidence_scale", str(param_value)])
    elif param_name == "neg_weight":
        cmd.extend(["--neg_weight", str(param_value)])
    elif param_name == "uncertainty_gamma":
        cmd.extend(["--uncertainty_gamma", str(param_value)])
    
    print(f"\n{'='*60}")
    print(f"Running: {param_name} = {param_value}")
    print(f"{'='*60}")
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Error running {param_name}={param_value}:")
            print(result.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout running {param_name}={param_value}")
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
                'parameter': param_name,
                'value': param_value,
                'best_score': metrics.get('Best Score', 0.0),
                'train_loss': metrics.get('Train Loss', 0.0),
                'val_loss': metrics.get('Val Loss', 0.0)
            }
    
    return None


def run_hyperparameter_analysis(dataset, output_dir):
    """
    Run hyperparameter sensitivity analysis.
    
    Args:
        dataset: Dataset name (odir or mured)
        output_dir: Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define hyperparameters to analyze
    hyperparameters = {
        'evidence_scale': [1.0, 5.0, 10.0, 15.0, 20.0],
        'neg_weight': [0.1, 0.2, 0.3, 0.5, 1.0],
        'uncertainty_gamma': [0.5, 1.0, 1.5, 2.0, 3.0]
    }
    
    print(f"Running hyperparameter sensitivity analysis for {dataset}")
    print(f"Output directory: {output_dir}")
    
    # Run all hyperparameter combinations
    all_results = []
    for param_name, param_values in hyperparameters.items():
        for param_value in param_values:
            result = run_single_hyperparameter(dataset, param_name, param_value, output_dir)
            if result:
                all_results.append(result)
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = output_dir / f"hyperparameter_sensitivity_{dataset}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("Hyperparameter Sensitivity Analysis Summary")
        print("="*60)
        
        for param_name in hyperparameters.keys():
            param_results = results_df[results_df['parameter'] == param_name]
            if not param_results.empty:
                print(f"\n{param_name}:")
                for _, row in param_results.iterrows():
                    print(f"  {row['value']}: {row['best_score']:.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sensitivity analysis')
    parser.add_argument('--dataset', type=str, required=True, choices=['odir', 'mured'],
                        help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run analysis
    run_hyperparameter_analysis(args.dataset, args.output_dir)


if __name__ == '__main__':
    main()
