# This script will hopefully merge all 3 inference scripts together in the near future
import argparse
from argparse import ArgumentParser
from pathlib import Path
import os
from enum import Enum

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image
from matplotlib import pyplot as plt

from src.models.CycleGAN.train_cycle_gan import CycleGAN
from src.models.FRAN.train_fran import FRAN
from src.models.diffusion.diffusion import DiffusionModel


class Gender(Enum):
    MALE = 'M'
    FEMALE = 'F'

    def __str__(self):
        return self.value


class ModelType(Enum):
    CYCLEGAN = 'cyclegan'
    FRANMODEL = 'fran'
    DIFFUSION = 'diffusion'

    def __str__(self):
        return self.value


def imsave(img):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    norm = (npimg - np.min(npimg)) / (np.max(npimg) - np.min(npimg))
    plt.imsave("genarray3.png", norm)


def generate(
        model_type: ModelType,
        ckpt_path: Path | str,
        input_img: Path | str,
        input_age: int,
        input_gender: Gender,
        save_path: str | Path = Path("examples/out"),
        gif: bool = False,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_image_tensor = read_image(input_img, mode=ImageReadMode.RGB) # C x H x W
    img_height = input_image_tensor.size(1)
    img_width = input_image_tensor.size(2)

    if model_type == ModelType.CYCLEGAN:
        resize_size = min(244, img_height, img_width)
        model = CycleGAN.load_from_checkpoint(ckpt_path)
    elif model_type == ModelType.DIFFUSION:
        resize_size = min(180, img_height, img_width)
        model = DiffusionModel.load_from_checkpoint(ckpt_path)
    elif model_type == ModelType.FRANMODEL:
        resize_size = min(512, img_width, img_height)
        model = FRAN.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError("Model type not implemented.")

    transform = transforms.Compose([
            transforms.ConvertImageDtype(dtype=torch.float),
            transforms.Resize((resize_size, resize_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    model.eval()
    if model_type != ModelType.FRANMODEL:
        input_image_tensor = transform(read_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_image_tensor)
        output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        min_val, max_val = output.min(), output.max()
        new_image_normalized = 255 * (output - min_val) / (max_val - min_val)
        new_image_normalized = new_image_normalized.astype('uint8')
        sample_image = Image.fromarray(new_image_normalized)
        sample_image.save(os.path.join(save_path, f"generated_{model_type.value}"), format='png')
    else:
        target_ages = [13, 18, 23, 33, 38, 48, 53, 58, 63, 68, 73, 78]
        input_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), input_age / 100)
        out_image_array = []
        for target_age in target_ages:
            target_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), target_age / 100)
            input_tensor = torch.cat((input_image_tensor, input_age_embedding, target_age_embedding), dim=0)
            with torch.no_grad():
                output = model(input_tensor.unsqueeze(0).to(device))

            new_image_out = output.squeeze(0).cpu()
            min_val, max_val = new_image_out.min(), new_image_out.max()
            new_image_normalized = 255 * (new_image_out - min_val) / (max_val - min_val)
            out_image_array.append(new_image_normalized)


def main():
    parser = ArgumentParser(description="Generate images from model.")
    parser.add_argument("--model_type", type=ModelType, help="Model type", required=True, choices=list(ModelType))
    parser.add_argument("--ckpt_path", type=Path, help="Path of model check  point to use.", required=True)
    parser.add_argument("--input_img", type=Path, help="Input image path.", required=True)
    parser.add_argument("--input_age", type=int, help="Input image age.", required=True)
    parser.add_argument("--input_gender", type=Gender, help="Input image gender", choices=list(Gender), required=True, default=Gender.MALE)
    parser.add_argument("--save", type=Path, help="Path of directory to save image to.", required=False)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, required=False, default=False)
    args = parser.parse_args()
    save_path = args.save if args.save is not None else Path("examples/out")
    is_gif = args.gif
    generate(args.model_type, args.ckpt_path, args.input_img, args.input_age, args.input_gender, save_path, is_gif)


if __name__ == "__main__":
    main()
