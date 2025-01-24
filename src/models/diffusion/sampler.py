import torch
import torch.nn as nn
import numpy as np


class DDPM:
    def __init__(self, beta_start: float = 1e-4, beta_end: float = 1e-2, n_timesteps: int = 1000):
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32) # linear schedule
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.n_timesteps = n_timesteps
        self.timesteps = torch.from_numpy(np.arange(0, n_timesteps)[::-1].copy())

    @torch.no_grad()
    def add_noise(self, original_sample: torch.FloatTensor, timestep: torch.IntTensor, noise: torch.Tensor) -> torch.FloatTensor:  # forward process
        alpha_bars = self.alpha_bars.to(device=original_sample.device, dtype=original_sample.dtype)
        timestep = timestep.to(device=original_sample.device)
        sqrt_alpha_bars = torch.sqrt(alpha_bars[timestep]) # mean
        sqrt_one_minus_alpha_bars = torch.sqrt((1.0 - alpha_bars[timestep]))  # stdev
        noisy_samples = (sqrt_alpha_bars * original_sample) + (sqrt_one_minus_alpha_bars) * noise
        return noisy_samples

    @torch.no_grad()
    def denoise_step(self, image: torch.Tensor, timestep: torch.IntTensor, predicted_noise: torch.Tensor) -> torch.Tensor:  # reverse process step
        # model output - noise predicted by the UNet
        t = timestep.to(device=image.device)
        betas = self.betas.to(device=image.device)
        beta_t = betas[t]
        alphas = self.alphas.to(device=image.device)
        alpha_t = alphas[t]
        alpha_bars = self.alpha_bars.to(device=image.device)
        alpha_bars_t = alpha_bars[t]


        pred_prev_sample = torch.pow(alpha_t, -0.5) * (image - beta_t / torch.sqrt(1 - alpha_bars_t) * predicted_noise)

        variance = 0
        if t > 0:
            device = image.device
            noise = torch.randn(image.shape, device=device, dtype=image.dtype)
            variance = torch.sqrt(self._get_variance(t)) * noise

        pred_prev_sample += variance
        return pred_prev_sample

    def _get_variance(self, timestep: torch.IntTensor) -> torch.Tensor:
        t = timestep
        prev_t = timestep - 1
        alpha_bars = self.alpha_bars.to(timestep.device)
        alpha_bars_t = alpha_bars[t]
        alpha_bars_t_prev = alpha_bars[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=timestep.device)
        curr_beta_t = 1.0 - alpha_bars_t / alpha_bars_t_prev

        variance = (1 - alpha_bars_t_prev) / (1 - alpha_bars_t) * curr_beta_t
        variance = torch.clamp(variance, min=1e-20)  # make sure it doesn't reach 0
        return variance