if __name__ == "__main__":
    import sys
    sys.path.append('../src')

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
import lightning as L # This is next, for now basic pytorch training loop
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

# from piq import LPIPS # maybe use lpips from here instead
from fran import Generator, Discriminator
from datasets.fgnet_loader import FGNETFRANDataset
from constants import FGNET_IMAGES_DIR

# FRAN model put together
class FRAN:
    def __init__(self, generator: Generator = None, discriminator: Discriminator = None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.generator = generator.to(device) if generator is not None else Generator(in_channels=5).to(device)
        self.discriminator = discriminator.to(device) if generator is not None else Discriminator(in_channels=4).to(device)


def prepare_cuda():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def prepare_transform():
    return transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((160, 160)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def visualize_images(input_images, aged_images, reconstruct=True, save=False, save_path=None):
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


def normalize_output(output: torch.tensor, min_val: float, max_val: float) -> torch.tensor:
    output = (output - min_val) / (max_val - min_val)
    return output * 2.0 - 1.0


def train():
    prepare_cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FRAN(device=device)
    for dl in model.generator.downsample:
        dl.to(device)
    for ul in model.generator.upsample:
        ul.to(device)
    l1_loss = nn.L1Loss()
    perceptual_loss = LPIPS(net_type='vgg').to(device)
    # perceptual_loss = LPIPS().to(device) - from piq package
    adversarial_loss = nn.BCEWithLogitsLoss()
    # as it was written (in the fran paper)
    lambda_l1 = 1.0
    lambda_lpips = 1.0
    lambda_adv = 0.05

    gen_optimizer = optim.Adam(model.generator.parameters(), lr=0.0001)
    dis_optimizer = optim.Adam(model.discriminator.parameters(), lr=0.0001)

    transform = prepare_transform()

    dataset = FGNETFRANDataset(FGNET_IMAGES_DIR, transform)
    n_valid_images = 16
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)

    num_epochs = 10

    for epoch in range(num_epochs):
        running_generator_loss = 0.0
        running_discriminator_loss = 0.0
        for i, batch in enumerate(train_loader):
            full_input = batch['input'].to(device)
            input_img = batch['input_img'].to(device)
            target_img = batch['target_img'].to(device)
            target_age = batch['target_age'].to(device)

            output = model.generator(full_input)
            predicted = input_img + output

            predicted = normalize_output(predicted, torch.min(predicted).item(), torch.max(predicted).item()) # maybe use different lpips

            predicted_with_age = torch.cat((input_img, target_age), dim=1)


            real = torch.ones(full_input.shape[0], 1, input_img.shape[2] // batch_size, input_img.shape[3] // batch_size).to(device)
            fake = torch.zeros(full_input.shape[0], 1, input_img.shape[2] // batch_size, input_img.shape[3] // batch_size).to(device)

            # Discriminator losses

            real_loss = adversarial_loss(model.discriminator(torch.cat((target_img, target_age), dim=1)), real)
            fake_loss = adversarial_loss(model.discriminator(predicted_with_age), fake)

            disc_loss = (real_loss + fake_loss) / 2

            disc_loss.backward()
            dis_optimizer.step()

            # Generator losses

            l1_loss_val = l1_loss(predicted, target_img)
            perceptual_loss_val = perceptual_loss(predicted, target_img)
            adversarial_loss_val = adversarial_loss(model.discriminator(predicted_with_age), real)

            gen_loss = lambda_l1 * l1_loss_val + lambda_lpips * perceptual_loss_val + lambda_adv * adversarial_loss_val

            gen_loss.backward()
            gen_optimizer.step()

            running_discriminator_loss += disc_loss.item()
            running_generator_loss += gen_loss.item()

        print(f"Epoch: [{epoch}/{num_epochs}] generator loss: {running_generator_loss / batch_size:.4f} discriminator loss: {running_discriminator_loss / batch_size:.4f} ")

    for i, batch in enumerate(valid_loader):
        input_img = batch['input'].to(device)
        out = age_images(model, input_img)
        visualize_images(input_img.cpu(), out, save=True)


def age_images(model, images):
    model.g.eval()
    with torch.no_grad():
        out = model.generator(images)
    return out.cpu()


if __name__ == "__main__":
    train()
