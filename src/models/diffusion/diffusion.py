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
                 sampler: DDPM | None = None,
                 lambda_cycle: float | None = None,
    ) -> None:
        super(DiffusionModel, self).__init__()
        self.g_a_b = generator if generator is not None else UNet()
        self.g_b_a = generator if generator is not None else UNet()
        self.diffusion_sampler = sampler if sampler is not None else DDPM()
        self.lambda_cycle = lambda_cycle
        self.diffusion_loss = nn.L1Loss()
        self.cycle_consistency_loss = nn.L1Loss()
        self.automatic_optimization = False

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        g_a_b_optimizer, g_b_a_optimizer = self.optimizers()

        g_a_b_optimizer.zero_grad()
        g_b_a_optimizer.zero_grad()

        x_a = batch["young_image"]
        x_b = batch["old_image"]
        batch_size = x_a.size(0)
        timestep = torch.randint(0, self.diffusion_sampler.n_timesteps (batch_size,), device=self.device)

        noisy_x_a = self.diffusion_sampler.add_noise(x_a, timestep)
        noisy_x_b = self.diffusion_sampler.add_noise(x_b, timestep)

        pred_noise_a = self.g_b_a(noisy_x_b, timestep)
        pred_noise_b = self.g_a_b(noisy_x_a, timestep)

        loss_ddpm_a = self.diffusion_loss(pred_noise_a, noisy_x_a)
        loss_ddpm_b = self.diffusion_loss(pred_noise_b, noisy_x_b)

        # cycle consistency
        reconstructed_x_a = self.diffusion_sampler.step(timestep, noisy_x_b, pred_noise_a)
        reconstructed_x_b = self.diffusion_sampler.step(timestep, noisy_x_a, pred_noise_b)

        cyc_loss = self.cycle_consistency_loss(x_a, reconstructed_x_a) + self.cycle_consistency_loss(x_b, reconstructed_x_b)
        loss = loss_ddpm_a + loss_ddpm_b + self.lambda_cycle * cyc_loss

        self.manual_backward(loss)
        g_a_b_optimizer.step()
        g_b_a_optimizer.step()

        # logging
        self.log_dict(
            {
                "loss_ddpm_a": loss_ddpm_a,
                "loss_ddpm_b": loss_ddpm_b,
                "cycle_consistency_loss": cyc_loss,
                "total_loss": loss,
            },
            prog_bar=True
        )
        return

    def configure_optimizers(self) -> list:
        g_a_b_optimizer = optim.Adam(self.g_a_b.parameters(), lr=1e-5, betas=(0.5, 0.999))
        g_b_a_optimizer = optim.Adam(self.g_b_a.parameters(), lr=1e-5, betas=(0.5, 0.999))
        return [g_a_b_optimizer, g_b_a_optimizer]
