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
        sqrt_one_minus_alpha_bars = torch.sqrt((1.0 - sqrt_alpha_bars[timestep]))  # stdev
        # equation (4) of the DDPM paper
        # X = mean + stdev * Z
        noisy_samples = (sqrt_alpha_bars * original_sample) + (sqrt_one_minus_alpha_bars) * noise
        return noisy_samples

    @torch.no_grad()
    def denoise_step(self, image: torch.Tensor, timestep: torch.IntTensor, predicted_noise: torch.Tensor) -> torch.Tensor:  # reverse process step
        # model output - noise predicted by the UNet
        t = timestep
        prev_t = timestep - 1
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bars_t = self.alpha_bars[t]

        pred_prev_sample = torch.pow(alpha_t, -0.5) * (image - beta_t / torch.sqrt(1 - alpha_bars_t) * predicted_noise)

        variance = 0
        if t > 0:
            device = image.device
            noise = torch.randn(image.shape, device=device, dtype=image.dtype)
            variance = torch.sqrt(self._get_variance(t)) * noise

        # reparametrization trick to go from N(0, 1) to N(mu, sigma) - same as in the VAE
        # X = mu + sigma * Z where Z ~ N(0, 1)
        pred_prev_sample += variance
        return pred_prev_sample

    def _get_variance(self, timestep: torch.IntTensor) -> torch.Tensor:
        t = timestep
        prev_t = timestep - 1
        alpha_bars_t = self.alpha_bars[t]
        alpha_bars_t_prev = self.alpha_bars[prev_t] if prev_t >= 0 else torch.Tensor(1.0)
        curr_beta_t = 1.0 - alpha_bars_t / alpha_bars_t_prev

        # according to formulas (6) and (7) in the DDPM paper
        variance = (1 - alpha_bars_t_prev) / (1 - alpha_bars_t) * curr_beta_t
        variance = torch.clamp(variance, min=1e-20)  # make sure it doesn't reach 0
        return variance