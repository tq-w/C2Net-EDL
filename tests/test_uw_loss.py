import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses.edl_loss import EDLMultiLabelLoss

def test_uw_weighting():
    gamma = 2.0
    loss_fn = EDLMultiLabelLoss(uncertainty_gamma=gamma)
    
    # Create controlled alpha/beta to test weighting
    # Case 1: alpha=10, beta=10 -> u = 2/(10+10) = 0.1 -> w = (1-0.1)^2 = 0.81
    # Case 2: alpha=2, beta=2 -> u = 2/(2+2) = 0.5 -> w = (1-0.5)^2 = 0.25
    alpha = torch.tensor([[10.0, 2.0]])
    beta = torch.tensor([[10.0, 2.0]])
    target = torch.tensor([[1.0, 1.0]])
    
    # Get total loss and components
    loss, components = loss_fn(alpha, beta, target, epoch=1) # Pass epoch to ensure annealing coef is calculated
    
    # Calculate likelihood loss manually for each element
    # p = alpha / (alpha + beta) = [0.5, 0.5]
    # For target=1: log_likelihood = digamma(alpha+beta) - digamma(alpha)
    s = alpha + beta
    expected_ll_elementwise = torch.digamma(s) - torch.digamma(alpha)
    
    # Expected weighted likelihood
    u = 2.0 / s
    w = torch.pow(1.0 - u, gamma)
    expected_weighted_ll = (expected_ll_elementwise * w).mean()
    
    # Check if the calculated likelihood_loss matches our expectation (with weighting)
    # The current implementation DOES NOT have weighting, so this should FAIL 
    # if it's currently just taking the mean of expected_ll_elementwise.
    
    current_unweighted_ll = expected_ll_elementwise.mean()
    print(f"\nExpected weighted LL: {expected_weighted_ll.item():.6f}")
    print(f"Current unweighted LL: {current_unweighted_ll.item():.6f}")
    print(f"Actual LL from component: {components['likelihood_loss']:.6f}")
    
    # If weighting is implemented, this should pass
    assert torch.allclose(torch.tensor(components['likelihood_loss']), expected_weighted_ll, atol=1e-5)
    print("Test passed!")

if __name__ == "__main__":
    try:
        test_uw_weighting()
    except AssertionError:
        print(f"Test failed as expected: Actual likelihood loss does not match weighted expectation.")
        exit(0) # Exit with 0 because failure is expected for TDD RED phase
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
