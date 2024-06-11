from itertools import pairwise

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from fgnet_loader import FGNETDataset
from cacd_loader import CACDDataset


class BasicUNet(nn.Module):
    def __init__(self, down_channels, up_channels):
        super(BasicUNet, self).__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2) for in_ch, out_ch in pairwise(down_channels)
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2) for in_ch, out_ch in pairwise(up_channels)
        ])
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.upscale = nn.MaxPool2d(2)
        self.downscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        residual_inputs = []
        for i, l in enumerate(self.down_layers):
            x = self.LeakyReLU(l(x)) # Through the layer and the activation function
            if i < len(self.down_layers) - 1: # For all but the third (final) down layer:
              residual_inputs.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += residual_inputs.pop() # Fetching stored output (skip connection)
            x = self.LeakyReLU(l(x)) # Through the layer and the activation function

        return x


def linear_beta_schedule(steps, start=0.0001, end=0.02):
    return torch.linspace(start, end, steps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion(x_0, t, steps=200, device="cpu"):
    sqrt_alphas_cumprod, _, sqrt_one_minus_alphas_cumprod, _ = pre_calc_terms(timesteps=steps)
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def pre_calc_terms(timesteps):
    betas = linear_beta_schedule(timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance


def main():
    net = BasicUNet(down_channels=(3, 32, 64, 128), up_channels=(128, 64, 32, 3))
    x = torch.rand(8, 3, 32, 32) # batch_size, channels, W, H)

    print(sum(p.numel() for p in net.parameters()))
    print(net(x)[0].shape)


if __name__ == "__main__":
    main()