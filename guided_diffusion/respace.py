"""
Spaced and multi‑stage diffusion utilities.

Changes w.r.t. the original implementation
------------------------------------------
1.  Build two O(1) lookup tables once:
        • timestep_to_index   : original t  -> spaced index
        • timestep_to_stage   : spaced index -> stage id
    (The old code searched linearly with `== … .nonzero()`.)

2.  `get_stage`, `p_mean_variance`, `training_losses` now use these
    tables directly – no more IndexError / empty nonzero.

3.  `stage_indices` are always put into `model_kwargs` under the same
    key (`"stage_indices"`) to keep the UNet API consistent.
"""

from __future__ import annotations

import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


# -----------------------------------------------------------------------------#
# helpers
# -----------------------------------------------------------------------------#


def space_timesteps(num_timesteps: int, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)



# -----------------------------------------------------------------------------#
# base class that skips steps from an underlying diffusion process
# -----------------------------------------------------------------------------#


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process that uses only a subset of the original timesteps.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map: list[int] = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        # ---------------- new ---------------- #
        # reverse look‑up: original timestep -> spaced index
        self.timestep_to_index = {t_orig: idx for idx, t_orig in enumerate(self.timestep_map)}
        # ------------------------------------- #

        kwargs["betas"] = np.array(new_betas, dtype=np.float64)
        super().__init__(**kwargs)

    # wrap / unwrap ----------------------------------------------------------------

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

    def condition_mean(self, cond_fn, *a, **kw):
        return super().condition_mean(self._wrap_model(cond_fn), *a, **kw)

    def condition_score(self, cond_fn, *a, **kw):
        return super().condition_score(self._wrap_model(cond_fn), *a, **kw)

    # -----------------------------------------------------------------------------#
    def _scale_timesteps(self, t):
        # scaling is handled by _WrappedModel
        return t


# -----------------------------------------------------------------------------#
# multi‑stage variant
# -----------------------------------------------------------------------------#


class MultiStageSpacedDiffusion(SpacedDiffusion):
    """
    Same as SpacedDiffusion but every spaced timestep is assigned to a stage
    (e.g. to choose the proper decoder).
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

        self.num_stages = num_stages
        N = len(self.timestep_map)

        # --------------- choose boundaries --------------- #
        if stage_distribution == "linear":
            step = N // num_stages
            self.stage_boundaries = [step * (i + 1) for i in range(num_stages - 1)] + [N]
        elif stage_distribution == "geometric":
            remaining, cur = N, 0
            self.stage_boundaries = []
            for _ in range(num_stages - 1):
                take = int(remaining * 0.6)
                cur += take
                self.stage_boundaries.append(cur)
                remaining -= take
            self.stage_boundaries.append(N)
        else:
            raise ValueError("unknown stage_distribution")

        # --------------- map spaced index -> stage id ---- #
        self.timestep_to_stage = {}
        prev = 0
        for stage_id, bound in enumerate(self.stage_boundaries):
            for idx in range(prev, bound):
                self.timestep_to_stage[idx] = stage_id
            prev = bound

    # helpers ---------------------------------------------------------------------

    def get_stage(self, spaced_idx: th.Tensor | np.ndarray | int):
        """
        Accepts spaced‑chain indices and returns stage id(s).
        """
        if isinstance(spaced_idx, th.Tensor):
            spaced_idx = spaced_idx.cpu().numpy()
        if np.isscalar(spaced_idx):
            return self.timestep_to_stage[int(spaced_idx)]
        return np.vectorize(self.timestep_to_stage.__getitem__)(spaced_idx)

    # override two methods --------------------------------------------------------

    def p_mean_variance(self, model, x, t, *args, **kw):
        kw = dict(kw or {})
        kw.setdefault("model_kwargs", {})
        # here `t` is already spaced‑index
        stages = th.as_tensor(self.get_stage(t), device=t.device, dtype=th.long)
        kw["model_kwargs"]["stage_indices"] = stages
        return super().p_mean_variance(model, x, t, *args, **kw)

    def training_losses(self, model, x_start, t, *args, **kw):
        kw = dict(kw or {})
        kw.setdefault("model_kwargs", {})
        stages = th.as_tensor(self.get_stage(t), device=t.device, dtype=th.long)
        kw["model_kwargs"]["stage_indices"] = stages
        return super().training_losses(model, x_start, t, *args, **kw)


# -----------------------------------------------------------------------------#
# internal wrapper (unchanged)
# -----------------------------------------------------------------------------#


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = th.as_tensor(timestep_map)
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        new_ts = self.timestep_map[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
