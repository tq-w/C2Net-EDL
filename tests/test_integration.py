import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses.edl_loss import EDLMultiLabelLoss, SupConLoss

def test_supcon_multilabel():
    print("Testing SupConLoss with multi-label targets...")
    batch_size = 4
    feat_dim = 128
    num_classes = 8
    
    loss_fn = SupConLoss(temperature=0.07)
    
    # Random normalized features
    features = torch.randn(batch_size, feat_dim)
    features = torch.nn.functional.normalize(features, dim=1)
    
    # Multi-label targets: samples 0 and 1 share label 0
    targets = torch.zeros(batch_size, num_classes)
    targets[0, 0] = 1
    targets[1, 0] = 1
    targets[2, 1] = 1
    targets[3, 2] = 1
    
    loss = loss_fn(features, targets)
    print(f"SupCon loss: {loss.item():.4f}")
    assert loss > 0
    print("SupConLoss test passed!")

def test_full_loss_integration():
    print("\nTesting EDL + SupCon integration logic...")
    # This simulates the logic in train_one_epoch_edl
    alpha = torch.randn(4, 8).abs() + 1.1
    beta = torch.randn(4, 8).abs() + 1.1
    targets = torch.randint(0, 2, (4, 8)).float()
    emb = torch.randn(4, 128)
    emb = torch.nn.functional.normalize(emb, dim=1)
    
    criterion = EDLMultiLabelLoss(uncertainty_gamma=1.0)
    criterion_contra = SupConLoss(temperature=0.07)
    
    loss_edl, components = criterion(alpha, beta, targets, epoch=1)
    loss_contra = criterion_contra(emb, targets)
    
    total_loss = loss_edl + 0.1 * loss_contra
    print(f"Total integrated loss: {total_loss.item():.4f}")
    assert total_loss > 0
    print("Integration test passed!")

if __name__ == "__main__":
    try:
        test_supcon_multilabel()
        test_full_loss_integration()
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
