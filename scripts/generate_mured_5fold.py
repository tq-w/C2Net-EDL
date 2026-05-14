#!/usr/bin/env python3
"""
Generate 5-fold cross-validation splits for MuReD dataset.

This script merges train_data.csv and val_data.csv from MuReD dataset,
then adds a 'fold' column for patient-level 5-fold cross validation.

Usage:
    python scripts/generate_mured_5fold.py \
        --train_csv /Users/wtq/PycharmProjects/DataProcess/mured/train_data.csv \
        --val_csv /Users/wtq/PycharmProjects/DataProcess/mured/val_data.csv \
        --output_csv datasets/MuReD_20class.csv \
        --n_folds 5 \
        --seed 42
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def extract_patient_id(sample_id):
    """
    Extract patient ID from sample ID.
    E.g., 'aria_c_25_1' -> 'aria_c_25', 'aria_d_9_3' -> 'aria_d_9'
    """
    parts = sample_id.split('_')
    # Remove the last part (image number)
    return '_'.join(parts[:-1])


def generate_5fold_splits(train_csv, val_csv, output_csv, n_folds=5, seed=42):
    """
    Generate patient-level 5-fold cross-validation splits.
    
    Args:
        train_csv: Path to train CSV file
        val_csv: Path to val CSV file
        output_csv: Path to output CSV file
        n_folds: Number of folds (default: 5)
        seed: Random seed for reproducibility
    """
    # Read input CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    print(f"Loaded {len(train_df)} training samples from {train_csv}")
    print(f"Loaded {len(val_df)} validation samples from {val_csv}")
    
    # Merge train and val
    df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Total samples after merging: {len(df)}")
    
    # Extract patient IDs
    df['patient_id'] = df['ID'].apply(extract_patient_id)
    
    # Get unique patients
    unique_patients = df['patient_id'].unique()
    n_patients = len(unique_patients)
    print(f"Found {n_patients} unique patients")
    
    # Set random seed
    np.random.seed(seed)
    
    # Shuffle patients
    shuffled_patients = np.random.permutation(unique_patients)
    
    # Assign folds to patients
    patient_fold_map = {}
    for i, patient_id in enumerate(shuffled_patients):
        patient_fold_map[patient_id] = i % n_folds
    
    # Map folds back to samples
    df['fold'] = df['patient_id'].map(patient_fold_map)
    
    # Verify fold distribution
    print("\nFold distribution:")
    for fold in range(n_folds):
        fold_df = df[df['fold'] == fold]
        n_samples = len(fold_df)
        n_patients = fold_df['patient_id'].nunique()
        print(f"  Fold {fold}: {n_samples} samples, {n_patients} patients")
    
    # Remove temporary patient_id column
    df = df.drop(columns=['patient_id'])
    
    # Save to output CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
    
    # Print sample of output
    print("\nSample of output:")
    print(df.head(10).to_string())
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate 5-fold CV splits for MuReD')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to train CSV file')
    parser.add_argument('--val_csv', type=str, required=True,
                        help='Path to val CSV file')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to output CSV file')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate splits
    generate_5fold_splits(args.train_csv, args.val_csv, args.output_csv, args.n_folds, args.seed)


if __name__ == '__main__':
    main()
