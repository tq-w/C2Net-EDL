"""
Optimizer utilities for layer-wise learning rate decay.

Contains:
- param_groups_lrd: Create parameter groups for layer-wise learning rate decay
"""

import json
import torch.nn as nn


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay.

    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58

    Args:
        model: PyTorch model
        weight_decay: Weight decay value for regularized parameters
        no_weight_decay_list: List of parameter names to exclude from weight decay
        layer_decay: Decay rate per layer

    Returns:
        param_groups: List of parameter group dictionaries
    """
    param_group_names = {}
    param_groups = {}

    if hasattr(model, 'blocks'):
        num_layers = len(model.blocks) + 1
    else:
        # Use ResNet layer count as default
        num_layers = (
            len(model.layer1) + len(model.layer2) + len(model.layer3) +
            len(model.layer4) + 1
        )

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # No decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id.

    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33

    Args:
        name: Parameter name
        num_layers: Total number of layers

    Returns:
        layer_id: Layer id for the parameter
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def get_params_groups(model):
    """
    Get parameter groups for optimizer.

    Separates regularized and non-regularized parameters.

    Args:
        model: PyTorch model

    Returns:
        param_groups: List of parameter group dictionaries
    """
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [
        {'params': regularized},
        {'params': not_regularized, 'weight_decay': 0.}
    ]


def has_batchnorms(model):
    """
    Check if model contains batch normalization layers.

    Args:
        model: PyTorch model

    Returns:
        has_bn: True if model has batch norm layers
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False