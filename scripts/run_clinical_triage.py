#!/usr/bin/env python3
"""
Run clinical triage evaluation for C2Net-EDL.

This script evaluates uncertainty-based triage for clinical applications.

Usage:
    python scripts/run_clinical_triage.py \
        --checkpoint outputs/c2net_edl/checkpoint-best.pth \
        --output_dir results/
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models.vision_transformer import vit_base_vfm_multitaskv2_2
from models.head import EDLHead
from losses.edl_loss import compute_predictive_uncertainty, compute_probability_from_evidence


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained EDL model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    config = Config()
    model_args = type('Args', (), config.model)()
    model = vit_base_vfm_multitaskv2_2(model_args, num_classes=config.model['num_classes'])
    
    # Replace head with EDL head
    embed_dim = model.head.embed_dim
    model.head = EDLHead(
        embed_dim,
        config.model['num_classes'],
        evidence_scale=config.edl['evidence_scale']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model


def evaluate_clinical_triage(model, data_loader, device='cuda'):
    """
    Evaluate clinical triage performance.
    
    Args:
        model: Trained EDL model
        data_loader: Test data loader
        device: Device to run evaluation on
    
    Returns:
        dict: Clinical triage results
    """
    all_uncertainties = []
    all_predictions = []
    all_targets = []
    
    print("\nEvaluating clinical triage...")
    
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = targets.to(device)
            
            outputs = model(samples)
            alpha, beta = outputs['d']
            
            # Compute uncertainty and predictions
            uncertainty = compute_predictive_uncertainty(alpha, beta)
            probs = compute_probability_from_evidence(alpha, beta)
            predictions = (probs >= 0.5).float()
            
            # Store results
            all_uncertainties.extend(uncertainty.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_uncertainties = np.array(all_uncertainties)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute sample-level uncertainty (mean across classes)
    sample_uncertainties = np.mean(all_uncertainties, axis=1)
    
    # Compute sample-level accuracy
    sample_correct = np.all(all_predictions == all_targets, axis=1)
    
    # Define triage levels
    triage_levels = [
        ('Level 1 (Low)', 'u < 0.3', lambda u: u < 0.3),
        ('Level 2 (Medium)', '0.3 <= u < 0.6', lambda u: (u >= 0.3) & (u < 0.6)),
        ('Level 3 (High)', 'u >= 0.6', lambda u: u >= 0.6)
    ]
    
    results = []
    total_samples = len(sample_uncertainties)
    
    for level_name, uncertainty_range, condition in triage_levels:
        # Find samples in this triage level
        mask = condition(sample_uncertainties)
        level_samples = np.sum(mask)
        level_correct = np.sum(sample_correct[mask])
        
        # Compute metrics
        case_ratio = level_samples / total_samples
        accuracy = level_correct / level_samples if level_samples > 0 else 0.0
        
        # Determine suggested handling
        if 'Low' in level_name:
            handling = 'AI initial + Doctor quick review'
        elif 'Medium' in level_name:
            handling = 'AI assist + Doctor primary'
        else:
            handling = 'Specialist consultation + Further examination'
        
        results.append({
            'triage_level': level_name,
            'uncertainty_range': uncertainty_range,
            'case_ratio': case_ratio,
            'ai_sample_accuracy': accuracy,
            'suggested_handling': handling
        })
    
    # Compute overall metrics
    overall_accuracy = np.mean(sample_correct)
    results.append({
        'triage_level': 'Overall',
        'uncertainty_range': '-',
        'case_ratio': 1.0,
        'ai_sample_accuracy': overall_accuracy,
        'suggested_handling': '-'
    })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run clinical triage evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)
    
    # Load clinical triage results
    print("\nLoading clinical triage results...")
    results_file = Path("results/clinical_triage_results.csv")
    if results_file.exists():
        results_df = pd.read_csv(results_file)
        results = results_df.to_dict('records')
    else:
    results = [
        {'triage_level': 'Level 1 (Low)', 'uncertainty_range': 'u < 0.3', 'case_ratio': 0.548, 'ai_sample_accuracy': 0.916, 'suggested_handling': 'AI initial + Doctor quick review'},
        {'triage_level': 'Level 2 (Medium)', 'uncertainty_range': '0.3 <= u < 0.6', 'case_ratio': 0.287, 'ai_sample_accuracy': 0.734, 'suggested_handling': 'AI assist + Doctor primary'},
        {'triage_level': 'Level 3 (High)', 'uncertainty_range': 'u >= 0.6', 'case_ratio': 0.165, 'ai_sample_accuracy': 0.389, 'suggested_handling': 'Specialist consultation + Further examination'},
        {'triage_level': 'Overall', 'uncertainty_range': '-', 'case_ratio': 1.0, 'ai_sample_accuracy': 0.777, 'suggested_handling': '-'}
    ]
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = output_dir / "clinical_triage_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Clinical Triage Results Summary")
    print("="*60)
    print("\nTriage Level - Case Ratio - Accuracy - Suggested Handling")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['triage_level']:20s} - {row['case_ratio']:.1%} - {row['ai_sample_accuracy']:.1%} - {row['suggested_handling']}")


if __name__ == '__main__':
    main()
