import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from cycle_gan import Discriminator, Generator
from fgnet_loader import FGNETCycleGANDataset


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
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def prep_optimizers(optim_params: list[tuple]):
    print(optim_params)
    optimizers = [optim.Adam(params=model.parameters(), lr=lr, betas=betas) for model, lr, betas in optim_params]
    return optimizers


# def collate_fn(batch):
#   return {
#       'young_image': torch.stack([x for x, _ in batch]),
#       'old_image': torch.tensor([y for _, y in batch])
# }


def train():
    images_path = "data/FGNET/FGNET/images"
    prepare_cuda()
    device = torch.device("cuda")

    # loss fn's
    adversarial_loss = nn.MSELoss()
    cycle_consistency_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    G = Generator().to(device) # young to old
    F = Generator().to(device) # old to young
    D_X = Discriminator(3).to(device)
    D_Y = Discriminator(3).to(device)

    optim_params = (0.0002, (0.5, 0.999))
    optimizer_G, optimizer_F, optimizer_D_X, optimizer_D_Y = prep_optimizers([(G, *optim_params), (F, *optim_params), (D_X, *optim_params), (D_Y, *optim_params)])

    transform = prepare_transform()

    dataset = FGNETCycleGANDataset(images_path, transform=transform)
    n_valid_images = 16
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_set = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)

    num_epochs = 5
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
            identity_loss_g = identity_loss(G(real_y), real_y) * 5.0 # lambda set to 5
            identity_loss_f = identity_loss(F(real_x), real_x) * 5.0

            # adversarial loss
            fake_y = G(real_x)
            fake_x = F(real_y)
            gan_loss_g = adversarial_loss(D_Y(fake_y), torch.ones_like(D_Y(fake_y)))
            gan_loss_f = adversarial_loss(D_X(fake_x), torch.ones_like(D_X(fake_x)))

            # cycle loss
            recovered_x = F(fake_y)
            recovered_y = G(fake_x)
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
            pred_real_x = D_X(real_x)
            pred_real_y = D_Y(real_y)
            loss_dx_real = adversarial_loss(pred_real_x, torch.ones_like(pred_real_x))
            loss_dy_real = adversarial_loss(pred_real_y, torch.ones_like(pred_real_y))

            # fake loss
            pred_fake_x = D_X(fake_x.detach())
            pred_fake_y = D_Y(fake_y.detach())
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
        print(f"Epoch: [{epoch}/{num_epochs}] loss G: {total_g_loss / batch_size:.4f} \
              loss F: {total_f_loss / batch_size:.4f} loss D_X {total_dx_loss / batch_size:4f} \
              loss D_Y {total_dy_loss / batch_size:.4f}")



if __name__ == "__main__":
    train()