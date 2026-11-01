from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnnealing


class _ParamContainer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = nn.ParameterList(list(params))


def _get_transformer_lr_mult_and_names(model):
    """
    Returns (lr_mult, transformer_param_names).
    Strict isolation: only active for RangeBEVFusion when fusion_method == 'transformer'.
    """
    target_module_name = None
    target_module = None
    for module_name, module in model.named_modules():
        if module.__class__.__name__ == 'RangeBEVFusion' and getattr(module, 'fusion_method', None) == 'transformer':
            target_module_name = module_name
            target_module = module
            break

    if target_module is None:
        return 1.0, set()

    transformer_cfg = target_module.model_cfg.get('TRANSFORMER_CONFIG', {})
    lr_mult = float(transformer_cfg.get('LR_MULT', 1.0))
    if abs(lr_mult - 1.0) < 1e-12:
        return 1.0, set()

    suffixes = (
        'token_proj.',
        'bev_token_proj.',
        'out_proj.',
        'transformer_attn.',
        'range_row_embed.',
        'range_col_embed.',
        'bev_row_embed.',
        'bev_col_embed.',
    )
    if target_module_name:
        prefixes = tuple(f'{target_module_name}.{x}' for x in suffixes)
        alpha_name = f'{target_module_name}.transformer_alpha'
    else:
        prefixes = suffixes
        alpha_name = 'transformer_alpha'

    names = set()
    for name, _ in model.named_parameters():
        if name == alpha_name or any(name.startswith(p) for p in prefixes):
            names.add(name)
    return lr_mult, names


def _split_base_and_transformer_params(model):
    lr_mult, transformer_names = _get_transformer_lr_mult_and_names(model)
    if len(transformer_names) == 0:
        return 1.0, [], []

    base_params, trans_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in transformer_names:
            trans_params.append(param)
        else:
            base_params.append(param)
    return lr_mult, base_params, trans_params


def build_optimizer(model, optim_cfg):
    lr_mult, base_params, trans_params = _split_base_and_transformer_params(model)
    use_transformer_scaling = len(trans_params) > 0 and abs(lr_mult - 1.0) > 1e-12

    if optim_cfg.OPTIMIZER == 'adam':
        if use_transformer_scaling:
            optimizer = optim.Adam([
                {'params': base_params, 'lr': optim_cfg.LR, 'weight_decay': optim_cfg.WEIGHT_DECAY},
                {'params': trans_params, 'lr': optim_cfg.LR * lr_mult, 'weight_decay': optim_cfg.WEIGHT_DECAY},
            ])
        else:
            optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'adamw':
        if use_transformer_scaling:
            optimizer = optim.AdamW([
                {'params': base_params, 'lr': optim_cfg.LR, 'weight_decay': optim_cfg.WEIGHT_DECAY},
                {'params': trans_params, 'lr': optim_cfg.LR * lr_mult, 'weight_decay': optim_cfg.WEIGHT_DECAY},
            ])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        if use_transformer_scaling:
            optimizer = optim.SGD([
                {'params': base_params, 'lr': optim_cfg.LR, 'weight_decay': optim_cfg.WEIGHT_DECAY},
                {'params': trans_params, 'lr': optim_cfg.LR * lr_mult, 'weight_decay': optim_cfg.WEIGHT_DECAY},
            ], momentum=optim_cfg.MOMENTUM)
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
                momentum=optim_cfg.MOMENTUM
            )
    elif optim_cfg.OPTIMIZER in ['adam_onecycle','adam_cosineanneal','adamw_onecycle','adamw_cosineanneal']:
        betas = optim_cfg.get('BETAS', (0.9, 0.99))
        betas = tuple(betas)
        optimizer_func = partial(optim.AdamW if ('adamw' in optim_cfg.OPTIMIZER) else optim.Adam, betas=betas)
        if use_transformer_scaling:
            # Create two logical layer groups so scheduler can keep a fixed lr ratio.
            layer_groups = [_ParamContainer(base_params), _ParamContainer(trans_params)]
            optimizer = OptimWrapper.create(
                optimizer_func, [3e-3, 3e-3], layer_groups, wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
            )
            optimizer.lr_mults = [1.0, lr_mult]
        else:
            def children(m: nn.Module):
                return list(m.children())

            def num_children(m: nn.Module) -> int:
                return len(children(m))

            flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
            get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
            optimizer = OptimWrapper.create(
                optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
            )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER in ['adam_onecycle', 'adamw_onecycle']:
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    elif optim_cfg.OPTIMIZER in ['adam_cosineanneal', 'adamw_cosineanneal']:
        lr_scheduler = CosineAnnealing(
            optimizer, total_steps, total_epochs, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.PCT_START, optim_cfg.WARMUP_ITER
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
