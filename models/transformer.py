from typing import NamedTuple

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


from fgnet_loader import FGNETDataset
from cacd_loader import CACDDataset


class VitParams(NamedTuple):
    img_size: int
    patch_size: int
    in_chans: int
    emb_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: int

class DiffusionParams(NamedTuple):
    steps: int=200
    beta_start: float=0.0001
    beta_end: float=0.02
    device: torch.device=torch.device("cuda")


class Diffusion:
    def __init__(self, steps, beta_start=0.0001, beta_end=0.02, device=torch.device("cuda")):
        self.betas = diffusion_linear_beta_schedule(steps, beta_start, beta_end)
        self.device = device

    def forward_diff(self, x_0, t):
        noise = torch.randn_like(x_0)
        alphas = 1 - self.betas
        alphas = torch.cumprod(alphas, dim=0).to(self.device)
        alphas = alphas[t]
        print(f"alphas: {alphas.to(torch.device("cpu")).shape}")
        print(f"x_0: {x_0.to(torch.device("cpu")).shape}")
        return torch.sqrt(alphas) * x_0 + torch.sqrt((1 - alphas)) * noise

    def reverse_diff(self, x_t, t):
        alphas = 1 - self.betas
        alphas = torch.cumprod(alphas, dim=0).to(self.device)
        alphas = alphas[t]
        return x_t / torch.sqrt(alphas)


class Embedding(nn.Module):
    def __init__(self, patch_size, in_chans, emb_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Reconstruct(nn.Module):
    def __init__(self, patch_size, emb_dim, out_chans):
        super().__init__()
        self.rec = nn.ConvTranspose2d(in_channels=emb_dim, out_channels=out_chans, kernel_size=patch_size, stride=patch_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.rec(x)
        x = self.relu(x)
        return x


class VITDiffusion(nn.Module):
    def __init__(self, vit_params: VitParams, diff_params: DiffusionParams):
        super().__init__()
        self.vit_params = vit_params
        self.patch_embedding = Embedding(vit_params.patch_size, vit_params.in_chans, vit_params.emb_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (vit_params.img_size // vit_params.patch_size) ** 2, vit_params.emb_dim))
        self.age_embedding = nn.Linear(1, vit_params.emb_dim)
        self.diffusion = Diffusion(diff_params.steps, diff_params.beta_start, diff_params.beta_end)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(vit_params.emb_dim, vit_params.num_heads, int(vit_params.emb_dim * vit_params.mlp_ratio)),
            vit_params.num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(vit_params.emb_dim, vit_params.num_heads, int(vit_params.emb_dim * vit_params.mlp_ratio)),
            vit_params.num_layers,
        )
        self.reconstruction = Reconstruct(vit_params.patch_size, vit_params.emb_dim, vit_params.in_chans)

    def forward(self, x, age, t):
        x = self.patch_embedding(x)
        x = x + self.pos_embedding

        age = age.unsqueeze(-1).float()
        age_embedding = self.age_embedding(age).unsqueeze(1)
        x = x + age_embedding
        encoded = self.encoder(x)
        tgt = torch.zeros_like(encoded)
        # x = self.diffusion.forward_diff(x, t)
        # x = self.diffusion.reverse_diff(x ,t)

        decoded = self.decoder(tgt, x)
        decoded = x.permute(0, 2, 1).view(-1, x.size(-1), self.vit_params.img_size // self.vit_params.patch_size, self.vit_params.img_size // self.vit_params.patch_size)
        output = self.reconstruction(decoded)
        return output


def diffusion_linear_beta_schedule(steps, start=0.0001, end=0.02):
    return torch.linspace(start, end, steps)


def prepare_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False



def train_model():
    device = torch.device("cuda")
    prepare_cuda()
    images_path = "FGNET/FGNET/individuals"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    batch_size = 16
    dataset = FGNETDataset(images_path, transform=transform)
    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

    vit_params = VitParams(img_size=256, patch_size=16, in_chans=3, emb_dim=768, num_layers=12, num_heads=12, mlp_ratio=4)
    diff_params = DiffusionParams(device=device)
    model = VITDiffusion(vit_params, diff_params).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for person, image, age, gender in iter(trainloader):
            timesteps = torch.zeros_like(age).to(device)
            image = image.to(device) # timestep 0 for now, implement properly later
            age = age.to(device)
            outputs = model(image, age, timesteps)
            loss = criterion(outputs, image)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * image.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


def main():
    train_model()

if __name__ == "__main__":
    main()
