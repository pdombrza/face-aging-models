if __name__ == "__main__":
    import sys
    sys.path.append('../src')

import copy
from datetime import datetime
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.constants import FGNET_IMAGES_DIR, CACD_META_SEX_ANNOTATED_PATH, CACD_SPLIT_DIR
from src.models.diffusion.sampler import DDPM
from src.models.diffusion.unet import UNet
from src.datasets.fgnet_loader import FGNETCycleGANDataset
from src.datasets.cacd_loader import CACDCycleGANDataset


class DiffusionModel(L.LightningModule):
    def __init__(self,
                 generator: UNet | None = None,
                 denoise_net: UNet | None = None,
                 sampler: DDPM | None = None,
                 lambda_cycle: float | None = None,
    ) -> None:
        super(DiffusionModel, self).__init__()
        self.g_a_b = generator if generator is not None else UNet(in_channels=3) # in_channels 3 because in generators 1 image
        self.g_b_a = copy.deepcopy(generator) if generator is not None else UNet(in_channels=3)
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
        print(x_a.shape)
        pred_x_a = self.g_b_a(x_b, torch.tensor([0], device=self.device))
        pred_x_b = self.g_a_b(x_a, torch.tensor([0], device=self.device))
        batch_size = x_a.size(0)
        timestep_a = torch.randint(0, self.diffusion_sampler.n_timesteps, (batch_size,), device=self.device)
        timestep_b = torch.randint(0, self.diffusion_sampler.n_timesteps, (batch_size,), device=self.device)
        noise_a = torch.randn(x_a.shape, device=self.device, dtype=x_a.dtype)
        noise_b = torch.randn(x_b.shape, device=self.device, dtype=x_b.dtype)
        noisy_x_a = []
        noisy_x_b = []
        noisy_pred_a = []
        noisy_pred_b = []
        for i, _ in enumerate(timestep_a):
            noisy_x_a.append(self.diffusion_sampler.add_noise(x_a[i], timestep_a[i], noise_a[i]))
            noisy_x_b.append(self.diffusion_sampler.add_noise(x_b[i], timestep_b[i], noise_b[i]))
            noisy_pred_a.append(self.diffusion_sampler.add_noise(pred_x_a[i], timestep_b[i], noise_a[i]))
            noisy_pred_b.append(self.diffusion_sampler.add_noise(pred_x_b[i], timestep_a[i], noise_b[i]))
        noisy_x_a = torch.stack(noisy_x_a, dim=0)
        noisy_x_b = torch.stack(noisy_x_b, dim=0)
        noisy_pred_a = torch.stack(noisy_pred_a, dim=0)
        noisy_pred_b = torch.stack(noisy_pred_b, dim=0)

        denoise_loss = self.denoiser_loss(noise_a, self.denoise_a(torch.cat([noisy_x_a, noisy_pred_b], 1), timestep_a)) + \
                        self.denoiser_loss(noise_b, self.denoise_b(torch.cat([noisy_x_b, noisy_pred_a], 1), timestep_b))

        dsm_loss = self.dsm_loss(noise_a, self.denoise_a(torch.cat([noisy_x_a, noisy_pred_b], 1), timestep_a)) + \
                    self.dsm_loss(noise_a, self.denoise_a(torch.cat([noisy_pred_a, noisy_x_b], 1), timestep_b)) + \
                    self.dsm_loss(noise_b, self.denoise_b(torch.cat([noisy_x_a, noisy_pred_a], 1), timestep_b)) + \
                    self.dsm_loss(noise_b, self.denoise_b(torch.cat([noisy_pred_b, noisy_x_a], 1), timestep_a))

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

    def forward(self, x_a: torch.Tensor) -> torch.Tensor: # inference
        x_b = torch.randn(x_a.shape, device=self.device, dtype=x_a.dtype)
        # arbitrary timestep necessary - half?
        t_r = self.diffusion_sampler.n_timesteps // 2
        with torch.no_grad():
            self.denoise_a.eval()
            self.denoise_b.eval()
            for t in range(self.diffusion_sampler.n_timesteps - 1, t_r, -1):
                timestep = torch.tensor([t])
                noise_a = torch.randn(x_a.shape, device=self.device, dtype=x_a.dtype)
                x_a = self.diffusion_sampler.add_noise(x_a, torch.tensor([t]), noise_a)
                x_b = self.diffusion_sampler.denoise_step(x_b, timestep, self.denoise_b(torch.cat([x_b, x_a], 1), torch.tensor([t], device=self.device)))

            for t in range(t_r, -1, -1):
                x_a = self.diffusion_sampler.denoise_step(x_a, timestep, self.denoise_a(torch.cat([x_a, x_b], 1), torch.tensor([t], device=self.device)))
                x_b = self.diffusion_sampler.denoise_step(x_b, timestep, self.denoise_b(torch.cat([x_b, x_a], 1), torch.tensor([t], device=self.device)))

        return x_b

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        if not self.logger:
            return

        with torch.no_grad():
            self.g_a_b.eval()
            self.g_b_a.eval()
            x_a = batch["young_image"]
            x_b = batch["old_image"]
            cycle_b = self.g_a_b(x_a, torch.tensor([0], device=self.device))
            cycle_a = self.g_b_a(x_b, torch.tensor([0], device=self.device))
            denoised_b = self.forward(x_a)

        self.logger.experiment.add_image("diff_cycle_b", torchvision.utils.make_grid(self._unnormalize_output(cycle_b)), self.current_epoch)
        self.logger.experiment.add_image("diff_cycle_a", torchvision.utils.make_grid(self._unnormalize_output(cycle_a)), self.current_epoch)
        self.logger.experiment.add_image("diffusion_denoised_b", torchvision.utils.make_grid(self._unnormalize_output(denoised_b)), self.current_epoch)

    def _unnormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 0.5) + 0.5


def main():
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((160, 160)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = FGNETCycleGANDataset(FGNET_IMAGES_DIR, transform)
    n_valid_images = 16
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)

    a = next(iter(train_loader))

    diffusion_model = DiffusionModel(lambda_cycle=10.0)

    # setup logging and checkpoints
    val_every_n_epochs = 1
    logger = TensorBoardLogger("tb_logs", "diffusion")
    checkpoint_callback = ModelCheckpoint(
        filename="diffusion_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,
    )

    diffusion_trainer = L.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=5,
        default_root_dir="../models/diffusion/",
        logger=logger,
        log_every_n_steps=val_every_n_epochs
        )
    diffusion_trainer.fit(diffusion_model, train_loader)

    diffusion_trainer.save_checkpoint(f"../models/diffusion_fin{datetime.now().strftime("%Y%m%d%H%M%S%z")}")


if __name__ == "__main__":
    main()