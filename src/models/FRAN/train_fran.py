import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from src.models.FRAN.fran import Generator, Discriminator
from src.models.FRAN.fran_utils import FRANLossLambdaParams


class FRAN(L.LightningModule):
    def __init__(
        self,
        generator: Generator | None = None,
        discriminator: Discriminator | None = None,
        loss_params: FRANLossLambdaParams | None = None,
    ) -> None:
        super(FRAN, self).__init__()
        self.generator = generator if generator is not None else Generator()
        self.discriminator = discriminator if discriminator is not None else Discriminator()
        self.loss_params = loss_params if loss_params is not None else FRANLossLambdaParams()
        self.l1_loss = nn.L1Loss()  # maybe dependency injection
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.perceptual_loss = LPIPS(net_type='vgg')
        self.automatic_optimization = False

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        gen_optimizer, dis_optimizer = self.optimizers()

        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        full_input = batch['input']
        input_img = batch['input_img']
        target_img = batch['target_img']
        target_age = batch['target_age']

        output = self.generator(full_input)
        predicted = input_img + output
        predicted = self._normalize_output(predicted, torch.min(predicted).item(), torch.max(predicted).item())
        predicted_with_age = torch.cat((predicted, target_age), dim=1)
        real_fake_loss_h = input_img.shape[2] // (2 ** self.discriminator.num_blocks)
        real_fake_loss_w = input_img.shape[3] // (2 ** self.discriminator.num_blocks)

        real = torch.ones(full_input.shape[0], 1, real_fake_loss_h, real_fake_loss_w).to(self.device)
        fake = torch.zeros(full_input.shape[0], 1, real_fake_loss_h, real_fake_loss_w).to(self.device)

        # Discriminator losses
        real_loss = self.adversarial_loss(self.discriminator(torch.cat((target_img, target_age), dim=1)), real)
        fake_loss = self.adversarial_loss(self.discriminator(predicted_with_age.detach()), fake)

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
        self.logger.experiment.add_image(f"output_{int(target_age[0][0][0][0].item() * 100)}", torchvision.utils.make_grid(self._unnormalize_output(predicted_norm[0])), self.current_epoch)
        self.logger.experiment.add_image(f"target_{int(target_age[0][0][0][0].item() * 100)}", torchvision.utils.make_grid(self._unnormalize_output(target_img[0])), self.current_epoch)

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

