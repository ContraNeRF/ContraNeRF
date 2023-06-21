import copy
from typing import Any, Dict, List, Optional, Set
import torch

from .lr_scheduler import WarmupMultiStepLR


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    custom_list = None,
):
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr

    overrides = {}
    if custom_list:
        for key, value in zip(custom_list[0], custom_list[1]):
            overrides[key] = {}
            for i in range(len(value) // 2):
                overrides[key][value[2*i]] = value[2*i+1]
    
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    # for module in model.modules():
        # for module_param_name, value in module.named_parameters(recurse=False):
    for module_param_name, value in model.named_parameters(recurse=True):
        if not value.requires_grad:
            continue
        # Avoid duplicating parameters
        if value in memo:
            continue
        memo.add(value)

        hyperparams = copy.copy(defaults)
        for key in overrides.keys():
            if key in module_param_name:
                hyperparams.update(overrides[key])
                break
        params.append({"params": [value], **hyperparams})
    return params


def build_optimizer(cfg, model):
    name = cfg.solver.optimizer
    base_lr = cfg.solver.base_lr
    custom_list = cfg.solver.custom_list
    params = get_default_optimizer_params(model, base_lr, custom_list)

    if name == "Adam":
        optim = torch.optim.Adam(params, lr=cfg.solver.base_lr)
    elif name == "AdamW":
        optim = torch.optim.AdamW(params, lr=cfg.solver.base_lr, weight_decay=cfg.solver.weight_decay)
    else:
        raise NotImplementedError
    return optim


def build_scheduler(cfg, optim):
    scheduler = WarmupMultiStepLR(
        optim, cfg.solver.milestones, cfg.solver.lr_decay_factor,
        cfg.solver.warmup_factor, cfg.solver.warmup_iters, cfg.solver.warmup_method,
    )
    return scheduler
