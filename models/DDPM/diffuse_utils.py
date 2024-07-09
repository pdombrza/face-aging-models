import torch

class DiffusionForward:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule="linear", cosine_s=8e-3):
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
            self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
            self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        # TODO implement cosine scheduling (but first obviously add image translation properties - BBDM)

    def add_noise(self, original, noise, t):
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.to(original.device)[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.to(original.device)[t]
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]
        return (sqrt_alpha_bar_t * original) + sqrt_one_minus_alpha_bar_t * noise
    

class DiffusionReverse:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule="linear", cosine_s=8e-3):
        self.b = torch.linspace(beta_start, beta_end, timesteps)
        self.a = 1 - self.b
        self.a_bar = torch.cumprod(self.a, dim=0)

    def sample_prev_timestamp(self, xt, noise_pred, t):
        self.a = self.a.to(xt.device)
        self.a_bar = self.a_bar.to(xt.device)
        self.b = self.b.to(xt.device)
        x0 = xt - (torch.sqrt(1 - self.a_bar[t]) * noise_pred)
        x0 = x0 / torch.sqrt(self.a_bar[t])
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = xt - ((1 - self.a[t]) * noise_pred) / (torch.sqrt(1 - self.a_bar[t]))
        mean = mean / torch.sqrt(self.a[t])

        if t == 0:
            return mean, x0
        
        else:
            variance = (1 - self.a_bar[t-1]) / (1 - self.a_bar[t])
            variance = variance * self.b
            sigma = variance ** 0.5
            z = torch.randn_like(xt).to(xt.device)
            return mean + sigma * z, x0


def get_time_embedding(timesteps, t_emb_dim):
    factor = 2 * torch.arange(start=0, end=t_emb_dim//2, dtype=torch.float32, device=timesteps.device) / t_emb_dim
    factor = 10000 ** factor
    t_emb = timesteps[:, None]
    t_emb = t_emb / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)
    return t_emb
