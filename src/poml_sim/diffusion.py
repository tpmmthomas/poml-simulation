"""Stable Diffusion DDPM recurrence with explicitly supplied per-step noise."""

from .protocol_inputs import gaussian_vector


def ddpm_step(sample, epsilon, alpha_t, alpha_previous, noise):
    """Compute the DDPM posterior mean and fixed-small variance update.

    Coefficients are derived from the scheduler's cumulative alphas, including
    skipped training timesteps. The final call uses alpha_previous=1 and no
    noise. Stable Diffusion latent predictions are not clipped to image range.
    """
    beta_t = 1 - alpha_t
    beta_previous = 1 - alpha_previous
    current_alpha = alpha_t / alpha_previous
    current_beta = 1 - current_alpha
    original = (sample - beta_t**0.5 * epsilon) / alpha_t**0.5
    mean = (alpha_previous**0.5 * current_beta / beta_t) * original
    mean = mean + (current_alpha**0.5 * beta_previous / beta_t) * sample
    if noise is None:
        return mean
    variance = max(float(beta_previous / beta_t * current_beta), 1e-20)
    return mean + variance**0.5 * noise


class DDPMModel:
    """Load SD v1-4 and expose denoiser-input states for the appendix experiment."""

    def __init__(
        self, model="CompVis/stable-diffusion-v1-4", *, device="cuda:0", steps=50, size=512
    ):
        import torch
        from diffusers import StableDiffusionPipeline, DDPMScheduler

        if not 2 <= steps <= 1000 or size < 64 or size % 8:
            raise ValueError("steps must be 2..1000; image size must be a multiple of 8")
        self.torch, self.device, self.steps, self.size = torch, device, steps, size
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=self.dtype).to(
            device
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config,
            clip_sample=False,
            variance_type="fixed_small",
            steps_offset=0,
        )
        self.scheduler.set_timesteps(steps)
        self.shape = (1, self.pipe.unet.config.in_channels, size // 8, size // 8)

    def noise(self, seed):
        """Expand one independent public challenge to a Gaussian latent tensor."""
        return self.torch.tensor(
            gaussian_vector(seed, self.torch.tensor(self.shape).prod().item()),
            dtype=self.dtype,
            device=self.device,
        ).reshape(self.shape)

    def trajectory(self, prompt, seeds, *, perturbation=None):
        """Return all T pre-denoiser states, using seeds[0] then fresh step noise."""
        torch = self.torch
        if len(seeds) != self.steps:
            raise ValueError("DDPM requires exactly T indexed randomness seeds")
        with torch.inference_mode():
            positive, negative = self.pipe.encode_prompt(prompt, self.device, 1, True)
            embeddings = torch.cat([negative, positive])
            state = self.noise(seeds[0])
            if perturbation is not None:
                state = state + perturbation
            states = []
            for index, timestep in enumerate(self.scheduler.timesteps):
                states.append(state.detach().float().cpu())
                predicted = self.pipe.unet(
                    torch.cat([state, state]),
                    timestep.to(self.device),
                    encoder_hidden_states=embeddings,
                ).sample
                unconditional, conditional = predicted.chunk(2)
                epsilon = unconditional + 7.5 * (conditional - unconditional)
                alpha = float(self.scheduler.alphas_cumprod[int(timestep)])
                last = index == self.steps - 1
                previous_alpha = (
                    1.0
                    if last
                    else float(
                        self.scheduler.alphas_cumprod[int(self.scheduler.timesteps[index + 1])]
                    )
                )
                state = ddpm_step(
                    state,
                    epsilon,
                    alpha,
                    previous_alpha,
                    None if last else self.noise(seeds[self.steps - 1 - index]),
                )
            return states
