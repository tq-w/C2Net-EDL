"""
Test script for C2Net EDL components.
"""

import torch
import numpy as np
from models.head import EDLHead
from losses.edl_loss import EDLLoss, kl_divergence, log_likelihood_loss
from utils.metrics import expected_calibration_error, uncertainty_analysis


def test_edl_head():
    """Test EDL head implementation."""
    print("Testing EDL Head...")
    embed_dim = 768
    num_classes = 8
    batch_size = 4

    head = EDLHead(embed_dim, num_classes)
    input_tensor = torch.randn(batch_size, embed_dim)

    alpha, beta = head(input_tensor)

    # Check shapes
    assert alpha.shape == (batch_size, num_classes), f"Alpha shape mismatch: {alpha.shape}"
    assert beta.shape == (batch_size, num_classes), f"Beta shape mismatch: {beta.shape}"

    # Check that parameters are > 1.0 (due to softplus + 1.0)
    assert torch.all(alpha > 1.0), "Alpha should be > 1.0"
    assert torch.all(beta > 1.0), "Beta should be > 1.0"

    # Test probability computation
    probs = head.predict_proba(alpha, beta)
    assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities should be in [0, 1]"

    # Test uncertainty computation
    uncertainty = head.compute_uncertainty(alpha, beta)
    assert torch.all(uncertainty > 0), "Uncertainty should be positive"

    print("✓ EDL Head tests passed!")


def test_edl_loss():
    """Test EDL loss implementation."""
    print("Testing EDL Loss...")
    batch_size = 4
    num_classes = 8

    # Create dummy evidence parameters (> 1.0)
    alpha = torch.rand(batch_size, num_classes) * 5 + 1.0
    beta = torch.rand(batch_size, num_classes) * 5 + 1.0
    target = torch.randint(0, 2, (batch_size, num_classes)).float()

    # Test KL divergence
    kl = kl_divergence(alpha, beta)
    assert kl.shape == (batch_size, num_classes), f"KL shape mismatch: {kl.shape}"
    assert torch.all(kl >= 0), "KL divergence should be non-negative"

    # Test log-likelihood loss
    ll_loss = log_likelihood_loss(alpha, beta, target)
    assert ll_loss.shape == (batch_size, num_classes), f"LL loss shape mismatch: {ll_loss.shape}"

    # Test EDL loss with annealing
    criterion = EDLLoss(kl_annealing_epochs=20)
    total_loss, components = criterion(alpha, beta, target, epoch=10)

    assert isinstance(total_loss, torch.Tensor), "Total loss should be a tensor"
    assert total_loss.item() > 0, "Total loss should be positive"
    assert components['annealing_coef'] == 0.5, f"Annealing coef should be 0.5, got {components['annealing_coef']}"

    print("✓ EDL Loss tests passed!")


def test_metrics():
    """Test EDL metrics implementation."""
    print("Testing EDL Metrics...")
    n_samples = 100
    n_classes = 8

    # Simulate EDL outputs
    alpha = np.random.rand(n_samples, n_classes) * 5 + 1.0
    beta = np.random.rand(n_samples, n_classes) * 5 + 1.0
    labels = np.random.randint(0, 2, (n_samples, n_classes))

    # Test ECE
    probs = alpha / (alpha + beta)
    ece = expected_calibration_error(probs, labels)
    assert ece >= 0, "ECE should be non-negative"
    assert ece <= 1.0, "ECE should be <= 1.0"

    # Test uncertainty analysis
    uncertainty_results = uncertainty_analysis(alpha, beta, labels)
    assert 'overall' in uncertainty_results, "Should have overall results"
    assert 'mean_uncertainty_correct' in uncertainty_results['overall'], "Should have correct uncertainty"
    assert 'mean_uncertainty_incorrect' in uncertainty_results['overall'], "Should have incorrect uncertainty"

    print("✓ EDL Metrics tests passed!")


def main():
    """Run all tests."""
    print("Running C2Net EDL component tests...\n")

    try:
        test_edl_head()
        test_edl_loss()
        test_metrics()
        print("\n🎉 All tests passed! C2Net EDL is ready to use.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == '__main__':
    main()