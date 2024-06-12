import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


from cycle_gan import Discriminator, Generator


def prepare_cuda():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def compute_loss():
    raise NotImplementedError


def prepare_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def prep_optimizers(optim_params: list[tuple]):
    print(optim_params)
    optimizers = [optim.Adam(params=model.parameters(), lr=lr, betas=betas) for model, lr, betas in optim_params]
    return optimizers


def train():
    prepare_cuda()
    device = torch.device("cuda")

    # loss fn's
    adversarial_loss = nn.MSELoss()
    cycle_consistency_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    G = Generator()
    F = Generator()
    D_X = Discriminator(3)
    D_Y = Discriminator(3)

    optim_params = (0.0002, (0.5, 0.999))
    optimizer_G, optimizer_F, optimizer_D_X, optimizer_D_Y = prep_optimizers([(G, *optim_params), (F, *optim_params), (D_X, *optim_params), (D_Y, *optim_params)])

    transforms = prepare_transforms()



if __name__ == "__main__":
    train()