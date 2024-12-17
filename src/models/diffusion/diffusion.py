import copy
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.diffusion.sampler import DDPM
from src.models.diffusion.unet import UNet


class DiffusionModel(L.LightningModule):
    def __init__(self,
                 generator: UNet | None = None,
                 denoise_net: UNet | None = None,
                 sampler: DDPM | None = None,
                 lambda_cycle: float | None = None,
    ) -> None:
        super(DiffusionModel, self).__init__()
        self.g_a_b = generator if generator is not None else UNet()
        self.g_b_a = copy.deepcopy(generator) if generator is not None else UNet()
        self.denoise_a = denoise_net if denoise_net is not None else UNet()
        self.denoise_b = copy.deepcopy(denoise_net) if denoise_net is not None else UNet()
        self.diffusion_sampler = sampler if sampler is not None else DDPM()
        self.lambda_cycle = lambda_cycle
        self.dsm_loss = nn.MSELoss()
        self.denoiser_loss = nn.MSELoss()
        self.cycle_consistency_loss = nn.L1Loss()
        self.automatic_optimization = False

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        g_a_b_optimizer, g_b_a_optimizer, denoise_a_optimizer, denoise_b_optimizer = self.optimizers()

        g_a_b_optimizer.zero_grad()
        g_b_a_optimizer.zero_grad()
        denoise_a_optimizer.zero_grad()
        denoise_b_optimizer.zero_grad()

        x_a = batch["young_image"]
        x_b = batch["old_image"]
        pred_x_a = self.g_b_a(x_b, torch.Tensor([0], device=self.device))
        pred_x_b = self.g_a_b(x_a, torch.Tensor([0], device=self.device))
        batch_size = x_a.size(0)
        timestep_a = torch.randint(0, self.diffusion_sampler.n_timesteps, (batch_size,), device=self.device)
        timestep_b = torch.randint(0, self.diffusion_sampler.n_timesteps, (batch_size,), device=self.device)

        noise_a = torch.randn(x_a.shape, device=self.device, dtype=x_a.dtype)
        noise_b = torch.randn(x_b.shape, device=self.device, dtype=x_b.dtype)

        noisy_x_a = self.diffusion_sampler.add_noise(x_a, timestep_a, noise_a)
        noisy_x_b = self.diffusion_sampler.add_noise(x_b, timestep_b, noise_b)
        noisy_pred_a = self.diffusion_sampler.add_noise(pred_x_a, timestep_b, noise_a)
        noisy_pred_b = self.diffusion_sampler.add_noise(pred_x_b, timestep_a, noise_b)

        denoise_loss = self.denoiser_loss(noise_a, self.denoise_a(torch.cat([noisy_x_a, noisy_pred_b, 1]), timestep_a)) + \
                        self.denoiser_loss(noise_b, self.denoise_b(torch.cat([noisy_x_b, noisy_pred_a, 1]), timestep_b))

        dsm_loss = self.dsm_loss(noise_a, self.denoise_a(torch.cat([noisy_x_a, noisy_pred_b, 1]), timestep_a)) + \
                    self.dsm_loss(noise_a, self.denoise_a(torch.cat([noisy_pred_a, noisy_x_b, 1]), timestep_b)) + \
                    self.dsm_loss(noise_b, self.denoise_b(torch.cat([noisy_x_a, noisy_pred_a, 1]), timestep_b)) + \
                    self.dsm_loss(noise_b, self.denoise_b(torch.cat([noisy_pred_b, noisy_x_a, 1]), timestep_a))

        cyc_loss = self.cycle_consistency_loss(x_a, self.g_b_a(pred_x_b)) + self.cycle_consistency_loss(x_b, self.g_a_b(pred_x_a))

        loss = dsm_loss + self.lambda_cycle * cyc_loss

        self.manual_backward(denoise_loss)
        self.manual_backward(loss)
        g_a_b_optimizer.step()
        g_b_a_optimizer.step()
        denoise_a_optimizer.step()
        denoise_b_optimizer.step()

        # logging
        self.log_dict(
            {
                "dsm_loss": dsm_loss,
                "denoise_loss": denoise_loss,
                "cycle_consistency_loss": cyc_loss,
                "dsm_loss+cyc_loss": loss,
            },
            prog_bar=True
        )
        return

    def configure_optimizers(self) -> list:
        g_a_b_optimizer = optim.Adam(self.g_a_b.parameters(), lr=1e-5, betas=(0.5, 0.999))
        g_b_a_optimizer = optim.Adam(self.g_b_a.parameters(), lr=1e-5, betas=(0.5, 0.999))
        denoise_a_optimizer = optim.Adam(self.denoise_a.parameters(), lr=1e-5, betas=(0.5, 0.999))
        denoise_b_optimizer = optim.Adam(self.denoise_b.parameters(), lr=1e-5, betas=(0.5, 0.999))
        return [g_a_b_optimizer, g_b_a_optimizer, denoise_a_optimizer, denoise_b_optimizer]

    def forward(self, x_a: torch.Tensor) -> torch.Tensor:
        x_b = torch.randn(x_a.shape, device=self.device, dtype=x_a.dtype)
        # arbitrary timestep necessary - half?
        t_r = self.diffusion_sampler.n_timesteps // 2
        for t in range(self.diffusion_sampler.n_timesteps, t_r, -1):
            timestep = torch.IntTensor([t], device=self.device)
            noise_a = torch.randn(x_a.shape, device=self.device, dtype=x_a.dtype)
            x_a = self.diffusion_sampler.add_noise(x_a, timestep, noise_a)
            x_b = self.diffusion_sampler.denoise_step(x_b, timestep, self.denoise_b(torch.cat([x_b, x_a], 1), timestep))

        for t in range(t_r, -1, -1):
            timestep = torch.IntTensor([t], device=self.device)
            x_a = self.diffusion_sampler.denoise_step(x_a, timestep, self.denoise_a(torch.cat([x_a, x_b], 1), timestep))
            x_a = self.diffusion_sampler.denoise_step(x_b, timestep, self.denoise_b(torch.cat([x_b, x_a], 1), timestep))

        return x_b
