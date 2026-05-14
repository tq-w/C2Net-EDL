#!/usr/bin/env python3
"""
Generate 5-fold cross-validation splits for ODIR 20-class dataset.

This script reads the ODIR_20class.csv file and adds a 'fold' column
for patient-level 5-fold cross validation.

Usage:
    python scripts/generate_odir_5fold.py \
        --input_csv /tmp/ODIR_20class_server.csv \
        --output_csv datasets/ODIR_20class.csv \
        --n_folds 5 \
        --seed 42
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def extract_patient_id(image_name):
    """
    Extract patient ID from image name.
    E.g., '0_left.jpg' -> '0', '12_right.jpg' -> '12'
    """
    return image_name.split('_')[0]


def generate_5fold_splits(input_csv, output_csv, n_folds=5, seed=42):
    """
    Generate patient-level 5-fold cross-validation splits.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        n_folds: Number of folds (default: 5)
        seed: Random seed for reproducibility
    """
    # Read input CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")
    
    # Extract patient IDs
    df['patient_id'] = df['image_name'].apply(extract_patient_id)
    
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
    parser = argparse.ArgumentParser(description='Generate 5-fold CV splits for ODIR')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV file')
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
    generate_5fold_splits(args.input_csv, args.output_csv, args.n_folds, args.seed)


if __name__ == '__main__':
    main()
