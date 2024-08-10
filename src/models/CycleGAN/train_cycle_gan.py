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

from cycle_gan import Discriminator, Generator
from constants import FGNET_IMAGES_DIR, CACD_META_SEX_ANNOTATED_PATH, CACD_SPLIT_DIR
from datasets.fgnet_loader import FGNETCycleGANDataset
from datasets.cacd_loader import CACDCycleGANDataset


class CycleGAN:
    # combines the generators and discriminators
    def __init__(self, generator=None, discriminator=None, device=torch.device("cuda")):
        self.g = generator.to(device) if generator is not None else Generator().to(device)
        self.f = generator.to(device) if generator is not None else Generator().to(device)
        self.d_x = discriminator.to(device) if discriminator is not None else Discriminator(3).to(device)
        self.d_y = discriminator.to(device) if discriminator is not None else Discriminator(3).to(device)



def prepare_cuda():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def compute_loss():
    raise NotImplementedError


def prepare_transform():
    return transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((160, 160)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def prep_optimizers(optim_params: list[tuple]):
    print(optim_params)
    optimizers = [optim.Adam(params=model.parameters(), lr=lr, betas=betas) for model, lr, betas in optim_params]
    return optimizers



def visualize_images(input_images, aged_images, reconstruct=True, save=False, save_path=None):
    if reconstruct:
        imgs = torch.stack([input_images, aged_images], dim=1).flatten(0,1)
        imgs = imgs / 2 + 0.5
        title = "Age progression"
        grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=False, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    if len(input_images) == 4:
        plt.figure(figsize=(10,10))
    else:
        plt.figure(figsize=(15,10))
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


def train():
    fgnet_images_path = FGNET_IMAGES_DIR
    prepare_cuda()
    device = torch.device("cuda")

    # loss fn's
    adversarial_loss = nn.MSELoss()
    cycle_consistency_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    # G = Generator().to(device) # young to old
    # F = Generator().to(device) # old to young
    # D_X = Discriminator(3).to(device)
    # D_Y = Discriminator(3).to(device)
    cycle_gan = CycleGAN()

    optim_params = (0.0002, (0.5, 0.999))
    optimizer_G, optimizer_F, optimizer_D_X, optimizer_D_Y = prep_optimizers([(cycle_gan.g, *optim_params), (cycle_gan.f, *optim_params), (cycle_gan.d_x, *optim_params), (cycle_gan.d_y, *optim_params)])

    transform = prepare_transform()

    cacd_meta = CACD_META_SEX_ANNOTATED_PATH
    cacd_images = CACD_SPLIT_DIR
    dataset = CACDCycleGANDataset(csv_file=cacd_meta, img_root_dir=cacd_images, transform=transform)
    n_valid_images = 16
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    batch_size = 4 # blows up my memory lmao
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_g_loss = 0.0
        total_f_loss = 0.0
        total_dx_loss = 0.0
        total_dy_loss = 0.0
        for i, (real_x, real_y) in enumerate(train_loader):
            real_x = real_x.to(device)
            real_y = real_y.to(device)

            ## generators ##
            optimizer_G.zero_grad()
            optimizer_F.zero_grad()

            # identity loss
            identity_loss_g = identity_loss(cycle_gan.g(real_y), real_y) * 5.0 # lambda set to 5
            identity_loss_f = identity_loss(cycle_gan.f(real_x), real_x) * 5.0

            # adversarial loss
            fake_y = cycle_gan.g(real_x)
            fake_x = cycle_gan.f(real_y)
            gan_loss_g = adversarial_loss(cycle_gan.d_y(fake_y), torch.ones_like(cycle_gan.d_y(fake_y)))
            gan_loss_f = adversarial_loss(cycle_gan.d_x(fake_x), torch.ones_like(cycle_gan.d_x(fake_x)))

            # cycle loss
            recovered_x = cycle_gan.f(fake_y)
            recovered_y = cycle_gan.g(fake_x)
            loss_cycle_x = cycle_consistency_loss(recovered_x, real_x) * 10.0
            loss_cycle_y = cycle_consistency_loss(recovered_y, real_y) * 10.0

            # total
            loss_g = identity_loss_g + gan_loss_g + loss_cycle_x
            loss_f = identity_loss_f + gan_loss_f + loss_cycle_y
            loss_g.backward()
            loss_f.backward()
            optimizer_G.step()
            optimizer_F.step()

            ## discriminators ##
            optimizer_D_X.zero_grad()
            optimizer_D_Y.zero_grad()

            # real loss
            pred_real_x = cycle_gan.d_x(real_x)
            pred_real_y = cycle_gan.d_y(real_y)
            loss_dx_real = adversarial_loss(pred_real_x, torch.ones_like(pred_real_x))
            loss_dy_real = adversarial_loss(pred_real_y, torch.ones_like(pred_real_y))

            # fake loss
            pred_fake_x = cycle_gan.d_x(fake_x.detach())
            pred_fake_y = cycle_gan.d_y(fake_y.detach())
            loss_dx_fake = adversarial_loss(pred_fake_x, torch.zeros_like(pred_fake_x))
            loss_dy_fake = adversarial_loss(pred_fake_y, torch.zeros_like(pred_fake_y))

            # total loss
            loss_dx = (loss_dx_real + loss_dx_fake) * 0.5
            loss_dy = (loss_dy_real + loss_dy_fake) * 0.5
            loss_dx.backward()
            loss_dy.backward()
            optimizer_D_X.step()
            optimizer_D_Y.step()

            total_g_loss += loss_g.item()
            total_f_loss += loss_f.item()
            total_dx_loss += loss_dx.item()
            total_dy_loss += loss_dy.item()
        print(f"Epoch: [{epoch}/{num_epochs}] loss cycle_gan.g: {total_g_loss / batch_size:.4f} loss cycle_gan.f: {total_f_loss / batch_size:.4f} loss cycle_gan.d_x {total_dx_loss / batch_size:.4f} loss cycle_gan.d_y {total_dy_loss / batch_size:.4f}")

    # evaluate
    for i, (real_x, real_y) in enumerate(valid_loader):
        real_x = real_x.to(device)
        out = age_images(cycle_gan, real_x)
        visualize_images(real_x.cpu(), out, save=True)
    # aged_images = age_images(cycle_gan, next(iter(valid_loader)))


def age_images(model, images):
    model.g.eval()
    with torch.no_grad():
        out = model.g(images)
    return out.cpu()


if __name__ == "__main__":
    train()