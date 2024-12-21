if __name__ == "__main__":
    import sys
    sys.path.append('../src')

from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from cycle_gan import Discriminator, Generator
from cycle_gan_utils import CycleGANLossLambdaParams
from constants import FGNET_IMAGES_DIR, CACD_META_SEX_ANNOTATED_PATH, CACD_SPLIT_DIR
from datasets.fgnet_loader import FGNETCycleGANDataset
from datasets.cacd_loader import CACDCycleGANDataset


class CycleGAN(L.LightningModule):
    # combines the generators and discriminators
    def __init__(
        self,
        optimizer_params: dict,
        generator: Generator | None = None,
        discriminator: Discriminator | None = None,
        loss_params: CycleGANLossLambdaParams | None = None,
    ) -> None:
        super(CycleGAN, self).__init__()
        self.g = generator if generator is not None else Generator()
        self.f = generator if generator is not None else Generator()
        self.d_x = discriminator if discriminator is not None else Discriminator()
        self.d_y = discriminator if discriminator is not None else Discriminator()
        self.loss_params = loss_params if loss_params is not None else CycleGANLossLambdaParams()
        self.adversarial_loss = nn.MSELoss()
        self.cycle_consistency_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.optimizer_params = optimizer_params
        self.automatic_optimization = False

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        real_x = batch["young_image"]
        real_y = batch["old_image"]

        g_optimizer, f_optimizer, d_x_optimizer, d_y_optimizer = self.optimizers()

        # Generators
        g_optimizer.zero_grad()
        f_optimizer.zero_grad()

        # identity loss
        identity_loss_g = self.identity_loss(self.g(real_y), real_y) * self.loss_params.lambda_identity  # lambda set to 5
        identity_loss_f = self.identity_loss(self.f(real_x), real_x) * self.loss_params.lambda_identity

        # adversarial loss
        fake_y = self.g(real_x)
        fake_x = self.f(real_y)
        gan_loss_g = self.adversarial_loss(self.d_y(fake_y), torch.ones_like(self.d_y(fake_y)))
        gan_loss_f = self.adversarial_loss(self.d_x(fake_x), torch.ones_like(self.d_x(fake_x)))

        # cycle loss
        recovered_x = self.f(fake_y)
        recovered_y = self.g(fake_x)
        loss_cycle_x = self.cycle_consistency_loss(recovered_x, real_x) * self.loss_params.lambda_cycle
        loss_cycle_y = self.cycle_consistency_loss(recovered_y, real_y) * self.loss_params.lambda_cycle

        # total
        loss_g = identity_loss_g + gan_loss_g + loss_cycle_x
        loss_f = identity_loss_f + gan_loss_f + loss_cycle_y
        self.manual_backward(loss_g)
        self.manual_backward(loss_f)
        g_optimizer.step()
        f_optimizer.step()

        # Discriminators
        d_x_optimizer.zero_grad()
        d_y_optimizer.zero_grad()

        # real loss
        pred_real_x = self.d_x(real_x)
        pred_real_y = self.d_y(real_y)
        loss_dx_real = self.adversarial_loss(pred_real_x, torch.ones_like(pred_real_x).to(self.device))
        loss_dy_real = self.adversarial_loss(pred_real_y, torch.ones_like(pred_real_y).to(self.device))

        # fake loss
        pred_fake_x = self.d_x(fake_x.detach())
        pred_fake_y = self.d_y(fake_y.detach())
        loss_dx_fake = self.adversarial_loss(pred_fake_x, torch.zeros_like(pred_fake_x).to(self.device))
        loss_dy_fake = self.adversarial_loss(pred_fake_y, torch.zeros_like(pred_fake_y).to(self.device))

        # total loss
        loss_dx = (loss_dx_real + loss_dx_fake) * self.loss_params.lambda_total
        loss_dy = (loss_dy_real + loss_dy_fake) * self.loss_params.lambda_total
        self.manual_backward(loss_dx)
        self.manual_backward(loss_dy)
        d_x_optimizer.step()
        d_y_optimizer.step()

        self.log_dict(
            {
                "cycle_gan_cycle_loss": loss_cycle_x + loss_cycle_y,
                "cycle_gan_adversarial_loss": gan_loss_g + gan_loss_f,
                "cycle_gan_identity_loss": identity_loss_g + identity_loss_f,
                "cycle_gan_total_g_loss": loss_g,
                "cycle_gan_total_f_loss": loss_f,
                "cycle_gan_total_dx_loss": loss_dx,
                "cycle_gan_total_dy_loss": loss_dy,
            },
            prog_bar=True
        )

        return

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        if not self.logger:
            return

        with torch.no_grad():
            real_x = batch["young_image"]
            real_y = batch["old_image"]
            fake_y = self.g(real_x)
            cycle_x = self.f(fake_y)
            fake_x = self.f(real_y)
            cycle_y = self.g(fake_x)

        self.logger.experiment.add_image("cycle_gan_fake_y", torchvision.utils.make_grid(self._unnormalize_output(fake_y)), self.current_epoch)
        self.logger.experiment.add_image("cycle_gan_fake_x", torchvision.utils.make_grid(self._unnormalize_output(fake_x)), self.current_epoch)
        self.logger.experiment.add_image("cycle_gan_cycle_y", torchvision.utils.make_grid(self._unnormalize_output(cycle_y)), self.current_epoch)
        self.logger.experiment.add_image("cycle_gan_cycle_x", torchvision.utils.make_grid(self._unnormalize_output(cycle_x)), self.current_epoch)

    def forward(self, x_img):
        with torch.no_grad():
            self.f.eval()
            generated_img = self.f(x_img)
        return generated_img

    def configure_optimizers(self):
        g_optimizer = optim.Adam(self.g.parameters(), **self.optimizer_params)
        f_optimizer = optim.Adam(self.f.parameters(), **self.optimizer_params)
        d_x_optimizer = optim.Adam(self.d_x.parameters(), **self.optimizer_params)
        d_y_optimizer = optim.Adam(self.d_y.parameters(), **self.optimizer_params)
        return [g_optimizer, f_optimizer, d_x_optimizer, d_y_optimizer]

    def _unnormalize_output(self, x):
        return (x * 0.5) + 0.5


def visualize_images(input_images: torch.Tensor, aged_images: torch.Tensor, reconstruct: bool = True, save: bool = False, save_path: bool = None) -> None:
    if reconstruct:
        imgs = torch.stack([input_images, aged_images], dim=1).flatten(0,1)
        imgs = imgs / 2 + 0.5
        title = "Age progression"
        grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=False, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    if len(input_images) == 4:
        plt.figure(figsize=(10, 10))
    else:
        plt.figure(figsize=(15, 10))
    plt.title(title)
    plt.imshow(grid)
    plt.axis('off')
    if save:
        if save_path is None:
            plt.savefig('temp_path.jpg')
        else:
            plt.savefig(save_path)
    else:
        plt.show()


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

    loss_params = CycleGANLossLambdaParams()
    optimizer_params = {"lr": 0.0002, "betas": (0.5, 0.999)}
    cycle_gan_model = CycleGAN(Generator(), Discriminator(), optimizer_params, loss_params)

    # setup logging and checkpoints
    val_every_n_epochs = 1
    logger = TensorBoardLogger("tb_logs", "cycle_gan")
    checkpoint_callback = ModelCheckpoint(
        filename="cycle_gan_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,
    )

    cycle_gan_trainer = L.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=5,
        default_root_dir="../models/cycle_gan/",
        logger=logger,
        log_every_n_steps=val_every_n_epochs
        )
    cycle_gan_trainer.fit(cycle_gan_model, train_loader, valid_loader)

    cycle_gan_trainer.save_checkpoint(f"../models/cycle_gan_fin{datetime.now().strftime("%Y%m%d%H%M%S%z")}")


if __name__ == "__main__":
    main()
