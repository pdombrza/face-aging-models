import torch
import torch.nn as nn
import numpy as np


class DDPM:
    def __init__(self, generator: torch.Generator, n_training_steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=torch.float32) ** 2  # huggingface scaled linear schedule
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.generator = generator
        self.n_training_steps = n_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, n_training_steps)[::-1].copy())

    def set_inference_steps(self, n_inference_steps: int = 50) -> None:
        self.n_inference_steps = n_inference_steps
        # 999, 998, ..., 0 => 1000 steps to 999, 979, 959, ... , 0 => 50 timesteps
        step_ratio = self.n_training_steps // self.n_inference_steps
        self.timesteps = torch.from_numpy((np.arange(0, n_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64))

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_bars = self.alpha_bars.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_bars = torch.sqrt(alpha_bars[timesteps])
        sqrt_alpha_bars = sqrt_alpha_bars.flatten()
        while len(sqrt_alpha_bars.shape) < len(original_samples.shape):
            sqrt_alpha_bars = sqrt_alpha_bars.unsqueeze(-1)

        sqrt_one_minus_alpha_bars = torch.sqrt((1.0 - sqrt_alpha_bars[timesteps]))  # stdev
        sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.flatten()
        while len(sqrt_one_minus_alpha_bars.shape) < len(sqrt_one_minus_alpha_bars.shape):
            sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.unsqueeze(-1)

        # equation (4) of the DDPM paper
        # X = mean + stdev * Z
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_bars * original_samples) + (sqrt_one_minus_alpha_bars) * noise
        return noisy_samples

    def step(self, timestep: int, latent_image: torch.Tensor, model_output: torch.Tensor):
        # model output - noise predicted by the UNet
        t = timestep
        prev_t = self._get_previous_timestep(t)
        alpha_bars_t = self.alpha_bars[t]
        alpha_bars_t_prev = self.alpha_bars[prev_t]
        beta_bars_t = 1.0 - alpha_bars_t
        beta_bars_t_prev = 1.0 - alpha_bars_t_prev
        curr_alpha_t = alpha_bars_t / alpha_bars_t_prev
        curr_beta_t = 1.0 - curr_alpha_t

        # predicted original sample - equation (15) of the DDPM paper
        pred_original_sample = (latent_image - torch.sqrt(beta_bars_t) * model_output) / torch.sqrt(alpha_bars_t)

        # coefficient for the pred_original_sample and
        pred_original_sample_coeff = torch.sqrt(alpha_bars_t_prev) * curr_beta_t / beta_bars_t
        # current sample x_t coefficient
        current_sample_coeff = torch.sqrt(curr_alpha_t) * beta_bars_t_prev / beta_bars_t

        # compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latent_image  # mean

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = torch.sqrt(self._get_variance(t)) * noise

        # reparametrization trick to go from N(0, 1) to N(mu, sigma) - same as in the VAE
        # X = mu + sigma * Z where Z ~ N(0, 1)
        pred_prev_sample += variance
        return pred_prev_sample

    def set_strength(self, strength: float=1.0) -> None:
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def _get_variance(self, timestep: int) -> torch.Tensor:
        t = timestep
        prev_t = self._get_previous_timestep(t)
        alpha_bars_t = self.alpha_bars[t]
        alpha_bars_t_prev = self.alpha_bars[prev_t] if prev_t >= 0 else torch.Tensor(1.0)
        curr_beta_t = 1.0 - alpha_bars_t / alpha_bars_t_prev

        # according to formulas (6) and (7) in the DDPM paper
        variance = (1 - alpha_bars_t_prev) / (1 - alpha_bars_t) * curr_beta_t
        variance = torch.clamp(variance, min=1e-20)  # make sure it doesn't reach 0
        return variance

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.n_training_steps / self.n_inference_steps
        return prev_t