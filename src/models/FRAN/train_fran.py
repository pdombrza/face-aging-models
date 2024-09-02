if __name__ == "__main__":
    import sys
    sys.path.append('../src')

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

# from piq import LPIPS # maybe use lpips from here instead
from fran import Generator, Discriminator
from datasets.fgnet_loader import FGNETFRANDataset
from datasets.cacd_loader import CACDFRANDataset
from constants import FGNET_IMAGES_DIR, CACD_META_SEX_ANNOTATED_PATH, CACD_SPLIT_DIR
from fran_utils import FRANLossLambdaParams


class FRAN(L.LightningModule):
    def __init__(self, generator: Generator, discriminator: Discriminator, loss_params: FRANLossLambdaParams) -> None:
        super(FRAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss_params = loss_params
        self.l1_loss = nn.L1Loss()  # maybe dependency injection
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.perceptual_loss = LPIPS(net_type='vgg')
        self.automatic_optimization = False

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        gen_optimizer, dis_optimizer = self.optimizers()
        full_input = batch['input']
        input_img = batch['input_img']
        target_img = batch['target_img']
        target_age = batch['target_age']


        output = self.generator(full_input)
        predicted = input_img + output
        predicted = self._normalize_output(predicted, torch.min(predicted).item(), torch.max(predicted).item())
        predicted_with_age = torch.cat((input_img, target_age), dim=1)
        real_fake_loss_h = input_img.shape[2] // (2 ** self.discriminator.num_blocks)
        real_fake_loss_w = input_img.shape[3] // (2 ** self.discriminator.num_blocks)

        real = torch.ones(full_input.shape[0], 1, real_fake_loss_h, real_fake_loss_w).to(self.device)
        fake = torch.zeros(full_input.shape[0], 1, real_fake_loss_h, real_fake_loss_w).to(self.device)

        # Discriminator losses
        real_loss = self.adversarial_loss(self.discriminator(torch.cat((target_img, target_age), dim=1)), real)
        fake_loss = self.adversarial_loss(self.discriminator(predicted_with_age), fake)

        disc_loss = (real_loss + fake_loss) / 2.0

        self.manual_backward(disc_loss)
        dis_optimizer.step()

        # Generator losses
        l1_loss_val = self.l1_loss(predicted, target_img)
        perceptual_loss_val = self.perceptual_loss(predicted, target_img)
        adversarial_loss_val = self.adversarial_loss(self.discriminator(predicted_with_age), real)

        gen_loss = self.loss_params.lambda_l1 * l1_loss_val + self.loss_params.lambda_lpips * perceptual_loss_val + self.loss_params.lambda_adv * adversarial_loss_val

        self.manual_backward(gen_loss)
        gen_optimizer.step()

        # Logging
        self.log_dict(
            {
                "fake_loss": fake_loss,
                "discriminator_loss": disc_loss,
                "generator_loss": gen_loss,
                "gen_adversarial_loss": adversarial_loss_val,
                "gen_perceptual_loss": perceptual_loss_val,
                "gen_l1_loss": l1_loss_val
            },
            prog_bar=True
        )

        return

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        if not self.logger:
            return

        with torch.no_grad():
            full_input = batch['input']
            input_img = batch['input_img']
            target_img = batch['target_img']
            target_age = batch['target_age']
            output = self.forward(full_input)

        predicted = input_img + output
        predicted_norm = self._normalize_output(predicted, torch.min(predicted).item(), torch.max(predicted).item())
        self.logger.experiment.add_image("input", torchvision.utils.make_grid(self._unnormalize_output(input_img[0])), self.current_epoch)
        self.logger.experiment.add_image(f"output_{target_age}", torchvision.utils.make_grid(self._unnormalize_output(predicted_norm[0])), self.current_epoch)
        self.logger.experiment.add_image(f"target_{target_age}", torchvision.utils.make_grid(self._unnormalize_output(target_img[0])), self.current_epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def configure_optimizers(self) -> list:
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
        dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        return [gen_optimizer, dis_optimizer]

    def _normalize_output(self, output: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        output = (output - min_val) / (max_val - min_val)
        return output * 2.0 - 1.0

    def _unnormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 0.5) + 0.5


def main():
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((160, 160)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    meta_path = CACD_META_SEX_ANNOTATED_PATH
    images_dir_path = CACD_SPLIT_DIR
    # dataset = CACDFRANDataset(meta_path, images_dir_path, transform=transform)
    dataset = FGNETFRANDataset(FGNET_IMAGES_DIR, transform)
    n_valid_images = 16
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)

    loss_params = FRANLossLambdaParams()
    fran_model = FRAN(Generator(in_channels=5), Discriminator(in_channels=4), loss_params)

    # setup logging and checkpoints
    val_every_n_epochs = 1
    logger = TensorBoardLogger("tb_logs", "fran")
    checkpoint_callback = ModelCheckpoint(
        filename="fran_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,
    )

    fran_trainer = L.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=15,
        max_time='00:24:00:00',
        default_root_dir="../models/fran/",
        logger=logger
        )
    fran_trainer.fit(fran_model, train_loader, valid_loader)
    fran_trainer.save_checkpoint("../models/fran_fin")


if __name__ == "__main__":
    main()
