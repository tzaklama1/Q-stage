# optimizer.py
"""
Optimizer factory using Optax
"""
from __future__ import annotations
from typing import Any, Optional
import optax

# --------------------------- utilities ------------------------------------- #
def _maybe_clip(clip_norm: Optional[float]) -> list[optax.GradientTransformation]:
    return [optax.clip_by_global_norm(clip_norm)] if (clip_norm and clip_norm > 0) else []


def _maybe_weight_decay(wd: float | None,
                        mask: Any | None) -> list[optax.GradientTransformation]:
    # Decoupled weight decay (AdamW-style). If `mask` is provided, decay only those params.
    if wd is None or wd <= 0:
        return []
    return [optax.add_decayed_weights(wd, mask=mask)]


def _warmup_cosine_schedule(peak_lr: float,
                            total_steps: int,
                            warmup_fraction: float = 0.06,
                            end_lr_scale: float = 0.1) -> optax.Schedule:
    """Linear warmup to `peak_lr`, then cosine decay to `end_lr_scale * peak_lr`."""
    warmup_steps = max(1, int(warmup_fraction * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(peak_lr),
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=float(end_lr_scale) * float(peak_lr),
    )


# --------------------------- main factory ---------------------------------- #
def create_optimizer(
    name: str,
    learning_rate: float | optax.Schedule,
    *,
    total_steps: Optional[int] = None,
    value_and_grad_func=None,
    **kwargs,
) -> optax.GradientTransformation:
    """
    Parameters
    ----------
    name : str
        One of:
        - "adam"                       : classic Adam
        - "adamw"                      : Adam with decoupled weight decay
        - "sgd_nesterov"               : SGD + momentum (Nesterov)
    learning_rate : float | optax.Schedule
        Scalar LR or schedule. For *_cosine_* and *_onecycle* variants, you can
        also pass a float and set `total_steps`.
    total_steps : int, optional
        Required for schedules that need the number of steps (cosine, onecycle).
    value_and_grad_func : callable, optional
        Only for "kfac": function mapping params->(loss, aux) to compute curvature.
    kwargs : dict
        Common optional keys (used by several optimizers):
        - clip_norm: float, global-norm gradient clipping (default: 1.0 for *_cosine_warmup)
        - weight_decay: float (AdamW, Lookahead wrapper, SGD via decoupled decay)
        - mask: PyTree mask for which params to apply weight decay to (e.g. exclude biases/LayerNorm)
        - b1, b2, eps: Adam-family hyperparameters
        - momentum: for SGD/Nesterov (default: 0.9)
        - nesterov: bool for SGD (default: True)
        - warmup_fraction, end_lr_scale: for warmup+cosine schedule
        - pct_start, div_factor, final_div_factor: for One-Cycle schedule
        - multiply_by_parameter_scale / clipping_threshold: Adafactor options
        - sync_period, slow_step_size: Lookahead options
        - For KFAC: damping, norm_constraint, value_func_has_aux
    """
    name = name.lower()
    wd = kwargs.get("weight_decay", None)
    mask = kwargs.get("mask", None)
    clip_norm = kwargs.get("clip_norm", None)
    b1 = kwargs.get("b1", 0.9)
    b2 = kwargs.get("b2", 0.95 if "adamw" in name else 0.999)  # slightly lower b2 is common for Transformers
    eps = kwargs.get("eps", 1e-8)

    # Allow float LR or schedule
    lr = learning_rate
    if isinstance(learning_rate, (int, float)) and name.endswith("cosine_warmup"):
        if total_steps is None:
            raise ValueError("total_steps is required for 'adamw_cosine_warmup'")
        lr = _warmup_cosine_schedule(
            peak_lr=float(learning_rate),
            total_steps=int(total_steps),
            warmup_fraction=kwargs.get("warmup_fraction", 0.06),
            end_lr_scale=kwargs.get("end_lr_scale", 0.1),
        )
        clip_norm = kwargs.get("clip_norm", 1.0) if clip_norm is None else clip_norm

    if name == "adam":
        return optax.chain(
            *_maybe_clip(clip_norm),
            optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps),
        )

    if name == "adamw":
        return optax.chain(
            *_maybe_clip(clip_norm),
            *_maybe_weight_decay(wd, mask),
            optax.adamw(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, weight_decay=3e-4),
        )

    if name == "sgd_nesterov":
        mom = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", True)
        return optax.chain(
            *_maybe_clip(clip_norm),
            *_maybe_weight_decay(wd, mask),
            optax.sgd(learning_rate=lr, momentum=mom, nesterov=nesterov),
        )

    raise ValueError(f"Unknown optimizer '{name}'.")