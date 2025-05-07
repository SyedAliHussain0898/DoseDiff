import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType



def space_timesteps(num_timesteps, section_counts):
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


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t

class MultiStageSpacedDiffusion(SpacedDiffusion):
    """
    A diffusion process that operates on multiple stages with different decoders.
    """
    def __init__(
        self,
        use_timesteps,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        num_stages=3,
        stage_distribution="linear",
    ):
        super().__init__(
            use_timesteps=use_timesteps,
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )
        
        # Define stage boundaries based on timesteps
        self.num_stages = num_stages
        
        if stage_distribution == "linear":
            # Divide timesteps evenly across stages
            self.stage_boundaries = []
            step_size = len(self.timestep_map) // num_stages
            for i in range(num_stages - 1):
                self.stage_boundaries.append((i + 1) * step_size)
            self.stage_boundaries.append(len(self.timestep_map))
        elif stage_distribution == "geometric":
            # Allocate more timesteps to early stages (higher noise levels)
            self.stage_boundaries = []
            remaining = len(self.timestep_map)
            ratio = 0.6  # Each subsequent stage gets 60% of remaining timesteps
            current = 0
            for i in range(num_stages - 1):
                stage_size = int(remaining * ratio)
                current += stage_size
                self.stage_boundaries.append(current)
                remaining -= stage_size
            self.stage_boundaries.append(len(self.timestep_map))
        
        # Create mapping from timestep to stage
        self.timestep_to_index = {orig_t: idx for idx, orig_t in enumerate(self.timestep_map)}
        prev_boundary = 0
        for i, boundary in enumerate(self.stage_boundaries):
            for t in range(prev_boundary, boundary):
                self.timestep_to_stage[t] = i
            prev_boundary = boundary
    
    def get_stage(self, t_idx):
        """
        Get the stage for the given timestep indices.
        """
        if isinstance(t_idx, th.Tensor):
            t_idx = t_idx.cpu().numpy()
        
        if isinstance(t_idx, (int, np.int64)):
            return self.timestep_to_stage[t_idx]
        else:
            return np.array([self.timestep_to_stage[idx] for idx in t_idx])
    
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t) with stage information.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # Convert timesteps to indices in the diffusion timesteps
        t_idx = torch.tensor(  
            [self.timestep_to_index[int(t_i)] for t_i in t.cpu().numpy()],
            device=t.device, dtype=torch.long
        )
        
        # Get the stage for each timestep
        stages = self.get_stage(t_idx)
        stages = th.tensor(stages, device=t.device, dtype=th.long)
        
        # Add stage information to model_kwargs
        model_kwargs["stages"] = stages
        
        # Call the original method with modified model_kwargs
        return super().p_mean_variance(model, x, t, clip_denoised, denoised_fn, model_kwargs)
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses with stage information.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # Convert timesteps to indices in the diffusion timesteps
         t_idx = torch.tensor(  
            [self.timestep_to_index[int(t_i)] for t_i in t.cpu().numpy()],
            device=t.device, dtype=torch.long
        )
        
        # Get the stage for each timestep
        stages = self.get_stage(t_idx)
        stages = th.tensor(stages, device=t.device, dtype=th.long)
        
        # Add stage information to model_kwargs
        model_kwargs["stage_indices"] = stages
        
        # Call the original method with modified model_kwargs
        return super().training_losses(model, x_start, t, model_kwargs, noise)


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
