import os

from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image
from matplotlib import pyplot as plt

from src.models.CycleGAN.cycle_gan import Generator
from src.models.CycleGAN.train_cycle_gan import CycleGAN
from generate_fran import imsave

CHECKPOINT_PATH = "models/cycle_gan/cycle_gan1/cycle_gan_fin"

def main():
    model = CycleGAN.load_from_checkpoint(CHECKPOINT_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_image = f'data/interim/synthetic_images_full/seed0002.png_23.jpg'
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((160, 160)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_image_tensor = transform(read_image(input_image, mode=ImageReadMode.RGB))

    model.eval()
    with torch.no_grad():
        output = model(input_image_tensor.unsqueeze(0).to(device))

    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    min_val, max_val = output.min(), output.max()
    new_image_normalized  = 255 * (output - min_val) / (max_val - min_val)
    new_image_normalized = new_image_normalized.astype('uint8')
    sample_image = Image.fromarray(new_image_normalized)
    sample_image.save(f"examples/output_image_cyclegan.png")

if __name__ == "__main__":
    main()
