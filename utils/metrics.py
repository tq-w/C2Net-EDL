"""
Evaluation metrics for multi-label classification with uncertainty quantification.

This implementation extends the original metrics with:
1. Expected Calibration Error (ECE) computation
2. Uncertainty analysis for correct vs incorrect predictions
3. Support for EDL-specific outputs (alpha, beta parameters)
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def calculate_metrics_per_class(true_labels, pred_labels, pred_sigmoids, writer, epoch, class_names=None):
    """
    Calculate evaluation metrics for each class.

    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted binary labels
        pred_sigmoids: Predicted probabilities
        writer: TensorBoard writer
        epoch: Current epoch number
        class_names: Optional list of class names

    Returns:
        metrics: Dictionary of per-class metrics
    """
    metrics = {}
    f1 = f1_score(true_labels, pred_labels, zero_division=0, average=None)
    roc = roc_auc_score(true_labels, pred_sigmoids, average=None)
    precision = precision_score(true_labels, pred_labels, zero_division=0, average=None)
    recall = recall_score(true_labels, pred_labels, zero_division=0, average=None)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    num_actual_classes = pred_sigmoids.shape[1] if len(pred_sigmoids.shape) > 1 else 1
    
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_actual_classes)]
    elif len(class_names) < num_actual_classes:
        class_names.extend([f'Class_{i}' for i in range(len(class_names), num_actual_classes)])
    elif len(class_names) > num_actual_classes:
        class_names = class_names[:num_actual_classes]

    for i, name in enumerate(class_names):
        if i >= true_labels.shape[1]:
            break

        true_labels_i = [row[i] for row in true_labels]
        pred_labels_i = [row[i] for row in pred_labels]

        accuracy = accuracy_score(true_labels_i, pred_labels_i)
        kappa = cohen_kappa_score(true_labels_i, pred_labels_i)
        score = (f1[i] + roc[i] + kappa) / 3

        metrics[i] = {
            'Accuracy': accuracy,
            'F1 Score': f1[i],
            'ROC AUC': roc[i],
            'Precision': precision[i],
            'Recall': recall[i],
            'Cohen\'s Kappa': kappa
        }
        print(
            f'Class {name}: Accuracy: {accuracy:.4f}, F1 Score: {f1[i]:.4f}, ROC AUC: {roc[i]:.4f}, '
            f'Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, Kappa: {kappa:.4f}, Score: {score:.4f}'
        )
        if writer is not None:
            writer.add_scalar(f'Precision/{name}', precision[i], epoch)
            writer.add_scalar(f'Recall/{name}', recall[i], epoch)
            writer.add_scalar(f'F1 Score/{name}', f1[i], epoch)
            writer.add_scalar(f'Accuracy/{name}', accuracy, epoch)
            writer.add_scalar(f'ROC AUC/{name}', roc[i], epoch)
            writer.add_scalar(f'Cohen\'s Kappa/{name}', kappa, epoch)
            writer.add_scalar(f'Score/{name}', score, epoch)

    return metrics


def expected_calibration_error(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.
    Lower ECE indicates better calibration.

    Args:
        probs: Predicted probabilities, shape (N,) or (N, num_classes)
        labels: True labels, shape (N,) or (N, num_classes)
        n_bins: Number of bins for calibration histogram

    Returns:
        ece: Expected calibration error value
    """
    probs = np.array(probs)
    labels = np.array(labels)

    if probs.ndim == 1:
        # Binary classification case
        probs = probs.reshape(-1, 1)
        labels = labels.reshape(-1, 1)

    num_classes = probs.shape[1]
    total_ece = 0.0

    for class_idx in range(num_classes):
        class_probs = probs[:, class_idx]
        class_labels = labels[:, class_idx]

        # Sort by predicted probability
        sorted_indices = np.argsort(class_probs)
        sorted_probs = class_probs[sorted_indices]
        sorted_labels = class_labels[sorted_indices]

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (sorted_probs > bin_lower) & (sorted_probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                # Average confidence in bin
                avg_confidence_in_bin = np.mean(sorted_probs[in_bin])
                # Accuracy in bin
                avg_accuracy_in_bin = np.mean(sorted_labels[in_bin])
                # Add to ECE
                ece += np.abs(avg_accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        total_ece += ece

    return total_ece / num_classes


def uncertainty_analysis(alpha, beta, labels, predictions=None, threshold=0.5):
    """
    Analyze uncertainty for correct vs incorrect predictions.

    Args:
        alpha: Alpha parameters from EDL model, shape (N, num_classes)
        beta: Beta parameters from EDL model, shape (N, num_classes)
        labels: True labels, shape (N, num_classes)
        predictions: Optional pre-computed predictions, shape (N, num_classes)
        threshold: Probability threshold for binary classification

    Returns:
        dict: Uncertainty analysis results
    """
    alpha = np.array(alpha)
    beta = np.array(beta)
    labels = np.array(labels)

    # Compute probabilities and uncertainty
    probs = alpha / (alpha + beta)
    uncertainty = 2.0 / (alpha + beta)

    # Compute predictions if not provided
    if predictions is None:
        predictions = (probs >= threshold).astype(np.float32)

    num_classes = probs.shape[1]
    results = {}

    for class_idx in range(num_classes):
        class_probs = probs[:, class_idx]
        class_uncertainty = uncertainty[:, class_idx]
        class_labels = labels[:, class_idx]
        class_predictions = predictions[:, class_idx]

        # Correct vs incorrect predictions
        correct_mask = (class_predictions == class_labels)
        incorrect_mask = ~correct_mask

        correct_uncertainty = class_uncertainty[correct_mask]
        incorrect_uncertainty = class_uncertainty[incorrect_mask]

        results[f'class_{class_idx}'] = {
            'mean_uncertainty_all': np.mean(class_uncertainty),
            'mean_uncertainty_correct': np.mean(correct_uncertainty) if len(correct_uncertainty) > 0 else 0.0,
            'mean_uncertainty_incorrect': np.mean(incorrect_uncertainty) if len(incorrect_uncertainty) > 0 else 0.0,
            'std_uncertainty_all': np.std(class_uncertainty),
            'std_uncertainty_correct': np.std(correct_uncertainty) if len(correct_uncertainty) > 0 else 0.0,
            'std_uncertainty_incorrect': np.std(incorrect_uncertainty) if len(incorrect_uncertainty) > 0 else 0.0,
            'num_correct': np.sum(correct_mask),
            'num_incorrect': np.sum(incorrect_mask)
        }

    # Overall statistics
    overall_correct_mask = np.all(predictions == labels, axis=1)
    overall_incorrect_mask = ~overall_correct_mask

    overall_uncertainty = np.mean(uncertainty, axis=1)
    overall_correct_uncertainty = overall_uncertainty[overall_correct_mask]
    overall_incorrect_uncertainty = overall_uncertainty[overall_incorrect_mask]

    results['overall'] = {
        'mean_uncertainty_all': np.mean(overall_uncertainty),
        'mean_uncertainty_correct': np.mean(overall_correct_uncertainty) if len(overall_correct_uncertainty) > 0 else 0.0,
        'mean_uncertainty_incorrect': np.mean(overall_incorrect_uncertainty) if len(overall_incorrect_uncertainty) > 0 else 0.0,
        'std_uncertainty_all': np.std(overall_uncertainty),
        'std_uncertainty_correct': np.std(overall_correct_uncertainty) if len(overall_correct_uncertainty) > 0 else 0.0,
        'std_uncertainty_incorrect': np.std(overall_incorrect_uncertainty) if len(overall_incorrect_uncertainty) > 0 else 0.0,
        'num_correct': np.sum(overall_correct_mask),
        'num_incorrect': np.sum(overall_incorrect_mask)
    }

    return results


def compute_edl_metrics(alpha, beta, labels, threshold=0.5, n_bins=10):
    """
    Compute comprehensive EDL metrics including calibration and uncertainty.

    Args:
        alpha: Alpha parameters from EDL model
        beta: Beta parameters from EDL model
        labels: True labels
        threshold: Probability threshold for binary classification
        n_bins: Number of bins for ECE computation

    Returns:
        dict: Comprehensive metrics dictionary
    """
    # Convert to numpy if needed
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu().numpy()
    if isinstance(beta, torch.Tensor):
        beta = beta.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Compute probabilities
    probs = alpha / (alpha + beta)
    predictions = (probs >= threshold).astype(np.float32)

    # Compute standard metrics
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Compute ROC AUC only for classes with both pos and neg samples to avoid NaN
    valid_auc_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            try:
                score_i = roc_auc_score(labels[:, i], probs[:, i])
                valid_auc_scores.append(score_i)
            except ValueError:
                pass
    
    if len(valid_auc_scores) > 0:
        roc_auc = np.mean(valid_auc_scores)
    else:
        roc_auc = np.nan

    # Compute ECE
    ece = expected_calibration_error(probs, labels, n_bins=n_bins)

    # Compute Brier Score: BS = (1/n) * Σ(y - p)^2
    brier_score = np.mean((labels - probs) ** 2)

    # Compute uncertainty analysis
    uncertainty_results = uncertainty_analysis(alpha, beta, labels, predictions, threshold)

    return {
        'f1_score': f1,
        'roc_auc': roc_auc,
        'ece': ece,
        'brier_score': brier_score,
        'uncertainty_analysis': uncertainty_results,
        'predictions': predictions,
        'probabilities': probs
    }


def log_uncertainty_analysis(results, writer=None, epoch=None, prefix=''):
    """
    Log uncertainty analysis results.

    Args:
        results: Results from uncertainty_analysis function
        writer: TensorBoard writer (optional)
        epoch: Current epoch (optional)
        prefix: Prefix for metric names (optional)
    """
    print(f"\n{prefix}Uncertainty Analysis Results:")
    print("=" * 50)

    # Overall results
    overall = results['overall']
    print(f"Overall:")
    print(f"  Mean uncertainty (all): {overall['mean_uncertainty_all']:.4f}")
    print(f"  Mean uncertainty (correct): {overall['mean_uncertainty_correct']:.4f}")
    print(f"  Mean uncertainty (incorrect): {overall['mean_uncertainty_incorrect']:.4f}")
    print(f"  Correct samples: {overall['num_correct']}")
    print(f"  Incorrect samples: {overall['num_incorrect']}")

    if writer is not None and epoch is not None:
        writer.add_scalar(f'{prefix}uncertainty/mean_all', overall['mean_uncertainty_all'], epoch)
        writer.add_scalar(f'{prefix}uncertainty/mean_correct', overall['mean_uncertainty_correct'], epoch)
        writer.add_scalar(f'{prefix}uncertainty/mean_incorrect', overall['mean_uncertainty_incorrect'], epoch)
        writer.add_scalar(f'{prefix}uncertainty/num_correct', overall['num_correct'], epoch)
        writer.add_scalar(f'{prefix}uncertainty/num_incorrect', overall['num_incorrect'], epoch)

    # Per-class results (show first few classes)
    class_keys = [k for k in results.keys() if k.startswith('class_')]
    for class_key in class_keys[:3]:  # Show first 3 classes
        class_result = results[class_key]
        print(f"\n{class_key}:")
        print(f"  Mean uncertainty (correct): {class_result['mean_uncertainty_correct']:.4f}")
        print(f"  Mean uncertainty (incorrect): {class_result['mean_uncertainty_incorrect']:.4f}")


if __name__ == '__main__':
    # Test the metrics functions
    np.random.seed(42)

    # Create dummy data
    n_samples = 1000
    n_classes = 8

    # Simulate EDL outputs
    alpha = np.random.rand(n_samples, n_classes) * 5 + 1.0
    beta = np.random.rand(n_samples, n_classes) * 5 + 1.0
    labels = np.random.randint(0, 2, (n_samples, n_classes))

    print("Testing EDL metrics:")
    print(f"Alpha shape: {alpha.shape}")
    print(f"Beta shape: {beta.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test ECE computation
    probs = alpha / (alpha + beta)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    print(f"ECE: {ece:.4f}")

    # Test uncertainty analysis
    uncertainty_results = uncertainty_analysis(alpha, beta, labels)
    print(f"Uncertainty analysis completed!")
    print(f"Overall mean uncertainty (correct): {uncertainty_results['overall']['mean_uncertainty_correct']:.4f}")
    print(f"Overall mean uncertainty (incorrect): {uncertainty_results['overall']['mean_uncertainty_incorrect']:.4f}")

    # Test comprehensive metrics
    metrics = compute_edl_metrics(alpha, beta, labels)
    print(f"\nComprehensive metrics:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")

    print("All tests passed!")