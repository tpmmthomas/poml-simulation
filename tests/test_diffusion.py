"""Check the explicit DDPM update against the upstream scheduler and metrics."""

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
from poml_sim.diffusion import ddpm_step


@pytest.mark.model
@pytest.mark.parametrize("step", [0, 1, 48, 49])
def test_ddpm_update_matches_official_scheduler_with_identical_noise(monkeypatch, step):
    scheduler = diffusers.DDPMScheduler(
        num_train_timesteps=1000, clip_sample=False, variance_type="fixed_small"
    )
    scheduler.set_timesteps(50)
    sample = torch.tensor([[0.4, -1.3, 0.7]])
    epsilon = torch.tensor([[-0.2, 0.1, 0.8]])
    noise = torch.tensor([[0.3, -0.9, 0.2]])
    monkeypatch.setattr("diffusers.schedulers.scheduling_ddpm.randn_tensor", lambda *a, **kw: noise)
    t = scheduler.timesteps[step]
    previous = scheduler.previous_timestep(t)
    actual = ddpm_step(
        sample,
        epsilon,
        float(scheduler.alphas_cumprod[t]),
        1.0 if previous < 0 else float(scheduler.alphas_cumprod[previous]),
        None if t == 0 else noise,
    )
    expected = scheduler.step(epsilon, t, sample).prev_sample
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
