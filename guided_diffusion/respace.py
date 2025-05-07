"""
Spaced and multi‑stage diffusion utilities.

 – Efficient lookup tables (no .nonzero())
 – Stage handling for multi‑decoder U‑Net
 – _WrappedModel implemented as nn.Module with a buffer
"""

from __future__ import annotations

import numpy as np
import torch as th
from torch import nn

from .gaussian_diffusion import GaussianDiffusion


# -----------------------------------------------------------------------------#
# helper: choose (spaced) timesteps
# -----------------------------------------------------------------------------#

def space_timesteps(num_timesteps: int, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            wanted = int(section_counts[4:])
            for stride in range(1, num_timesteps):
                if len(range(0, num_timesteps, stride)) == wanted:
                    return set(range(0, num_timesteps, stride))
            raise ValueError("cannot find integer stride for DDIM")
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra    = num_timesteps %  len(section_counts)
    start    = 0
    result   = []

    for i, cnt in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < cnt:
            raise ValueError(f"cannot divide section of {size} into {cnt}")
        stride = 1 if cnt <= 1 else (size - 1) / (cnt - 1)
        cur = 0.0
        for _ in range(cnt):
            result.append(start + round(cur))
            cur += stride
        start += size
    return set(result)


# -----------------------------------------------------------------------------#
# basic spaced diffusion
# -----------------------------------------------------------------------------#

class SpacedDiffusion(GaussianDiffusion):
    """A diffusion process that retains only a subset of the original steps."""

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map  = []                    # spaced‑idx -> original‑t
        self.original_num_steps = len(kwargs["betas"])

        base = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for t, ac in enumerate(base.alphas_cumprod):
            if t in self.use_timesteps:
                new_betas.append(1 - ac / last_alpha_cumprod)
                last_alpha_cumprod = ac
                self.timestep_map.append(t)

        # reverse lookup table: original‑t -> spaced‑idx
        self.timestep_to_index = {t: i for i, t in enumerate(self.timestep_map)}

        kwargs["betas"] = np.asarray(new_betas, dtype=np.float64)
        super().__init__(**kwargs)

    # --------------------------------------------------------------------- API

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model,
            self.timestep_map,
            self.rescale_timesteps,
            self.original_num_steps,
        )

    def p_mean_variance(self, model, *a, **kw):
        return super().p_mean_variance(self._wrap_model(model), *a, **kw)

    def training_losses(self, model, *a, **kw):
        return super().training_losses(self._wrap_model(model), *a, **kw)

    def condition_mean (self, fn, *a, **kw): return super().condition_mean (self._wrap_model(fn), *a, **kw)
    def condition_score(self, fn, *a, **kw): return super().condition_score(self._wrap_model(fn), *a, **kw)

    def _scale_timesteps(self, t):   # overridden: wrapped model does scaling
        return t


# -----------------------------------------------------------------------------#
# multi‑stage version (for multiple decoders)
# -----------------------------------------------------------------------------#

class MultiStageSpacedDiffusion(SpacedDiffusion):
    """
    Extends SpacedDiffusion by partitioning spaced steps into `num_stages`
    (e.g. to select different decoders).
    """

    def __init__(
        self,
        *,
        use_timesteps,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        num_stages: int = 3,
        stage_distribution: str = "linear",
    ):
        super().__init__(
            use_timesteps=use_timesteps,
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )

        N = len(self.timestep_map)
        self.num_stages = num_stages

        if stage_distribution == "linear":
            step = N // num_stages
            self.stage_boundaries = [step * (i + 1) for i in range(num_stages - 1)] + [N]
        elif stage_distribution == "geometric":
            remain, cur = N, 0
            self.stage_boundaries = []
            for _ in range(num_stages - 1):
                take = int(remain * 0.6)
                cur += take
                self.stage_boundaries.append(cur)
                remain -= take
            self.stage_boundaries.append(N)
        else:
            raise ValueError("unknown stage_distribution")

        # spaced‑idx -> stage‑id
        self.timestep_to_stage = {}
        prev = 0
        for sid, bound in enumerate(self.stage_boundaries):
            for idx in range(prev, bound):
                self.timestep_to_stage[idx] = sid
            prev = bound

    # helper -----------------------------------------------------------------

    def _stages_for_indices(self, spaced_idx: th.Tensor):
        cpu_idx = spaced_idx.detach().cpu().numpy()
        stage_np = np.vectorize(self.timestep_to_stage.__getitem__)(cpu_idx)
        return th.as_tensor(stage_np, device=spaced_idx.device, dtype=th.long)

    # overriden API ----------------------------------------------------------

    def p_mean_variance(self, model, x, t, *args, **kw):
        kw = dict(kw or {})
        kw.setdefault("model_kwargs", {})
        kw["model_kwargs"]["stage_indices"] = self._stages_for_indices(t)
        return super().p_mean_variance(model, x, t, *args, **kw)

    def training_losses(self, model, x_start, t, *args, **kw):
        kw = dict(kw or {})
        kw.setdefault("model_kwargs", {})
        kw["model_kwargs"]["stage_indices"] = self._stages_for_indices(t)
        return super().training_losses(model, x_start, t, *args, **kw)


# -----------------------------------------------------------------------------#
# Internal model wrapper
# -----------------------------------------------------------------------------#

class _WrappedModel(nn.Module):
    """
    Converts spaced‑chain indices back to original timesteps before calling the
    real UNet.  The lookup tensor is a *buffer* so it follows the model’s
    device / dtype and is saved in checkpoints automatically.
    """

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        super().__init__()
        self.model              = model
        self.rescale_timesteps  = rescale_timesteps
        self.original_num_steps = original_num_steps
        self.register_buffer("timestep_map",
                             th.tensor(timestep_map, dtype=th.long))

    def forward(self, x, ts, **kwargs):
        new_ts = self.timestep_map[ts]  # already on same device as model
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
