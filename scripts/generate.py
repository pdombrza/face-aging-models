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
from src.models.FRAN.fran import Generator as UnetGenerator
from src.models.diffusion.diffusion import DiffusionModel


class ModelType(Enum):
    CYCLEGAN = 'cyclegan'
    FRANMODEL = 'fran'
    DIFFUSION = 'diffusion'

    def __str__(self):
        return self.value


class DiffusionGenerator(Enum):
    CycleGAN = 'cyclegan'
    FranUNET = 'fran'

    def __str__(self):
        return self.value

def imsave(img, fname):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    norm = (npimg - np.min(npimg)) / (np.max(npimg) - np.min(npimg))
    plt.imsave(fname, norm)


def generate(
        model_type: ModelType,
        ckpt_path: Path | str,
        input_img:  str,
        img_size: int | None = None,
        fran_input_age: int | None = None,
        save_path: str | Path = Path("examples/out"),
        fran_target_age: int | None = None,
        cyclegan_reverse: bool = False,
        diffusion_translator: DiffusionGenerator | None = None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_image_tensor = read_image(input_img, mode=ImageReadMode.RGB) # C x H x W
    img_height = input_image_tensor.size(1)
    img_width = input_image_tensor.size(2)

    if model_type == ModelType.CYCLEGAN:
        image_size = img_size if img_size is not None else 244
        resize_size = min(image_size, img_height, img_width)
        model = CycleGAN.load_from_checkpoint(ckpt_path)
    elif model_type == ModelType.DIFFUSION:
        image_size = img_size if img_size is not None else 180
        resize_size = min(image_size, img_height, img_width)
        if diffusion_translator == DiffusionGenerator.CycleGAN:
            model = DiffusionModel.load_from_checkpoint(ckpt_path)
        else:
            model = DiffusionModel.load_from_checkpoint(ckpt_path, generator=UnetGenerator(in_channels=3))
    elif model_type == ModelType.FRANMODEL:
        image_size = img_size if img_size is not None else 512
        resize_size = min(512, img_width, img_height)
        model = FRAN.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError("Model type not implemented.")

    transform = transforms.Compose([
            transforms.ConvertImageDtype(dtype=torch.float),
            transforms.Resize((resize_size, resize_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    model.to(device)
    model.eval()
    if model_type != ModelType.FRANMODEL:
        input_tensor = transform(input_image_tensor)
    else:
        if fran_target_age is None:
            raise ValueError("Target age required for FRAN model.")
        if fran_target_age is None:
            raise ValueError("Target age required for FRAN model.")
        input_image_tensor = transform(input_image_tensor)
        input_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), fran_input_age / 100)
        target_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), fran_target_age / 100)
        input_tensor = torch.cat((input_image_tensor, input_age_embedding, target_age_embedding), dim=0)
    with torch.no_grad():
        if model_type == ModelType.CYCLEGAN:
            output = model(input_tensor.unsqueeze(0).to(device), reverse=cyclegan_reverse)
        else:
            output = model(input_tensor.unsqueeze(0).to(device))
    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    min_val, max_val = output.min(), output.max()
    new_image_normalized = 255 * (output - min_val) / (max_val - min_val)
    new_image_normalized = new_image_normalized.astype('uint8')
    sample_image = Image.fromarray(new_image_normalized)
    sample_image.save(os.path.join(save_path, f"generated_{model_type.value}.png"))


def main():
    parser = ArgumentParser(description="Generate images from model.")
    parser.add_argument("--model_type", type=ModelType, help="Model type", required=True, choices=list(ModelType))
    parser.add_argument("--ckpt_path", type=Path, help="Path of model checkpoint to use.", required=True)
    parser.add_argument("--input_img", type=str, help="Input image path.", required=True)
    parser.add_argument("--img_size", type=int, help="Input image size.", required=False)
    parser.add_argument("--fran_input_age", type=int, help="Input image age for FRAN.", required=False)
    parser.add_argument("--save", type=Path, help="Path of directory to save image to.", required=False)
    parser.add_argument("--fran_target_age", type=int, help="Target age for FRAN model. Not required when using other models.", required=False)
    parser.add_argument("--cyclegan_reverse", default=False, action=argparse.BooleanOptionalAction, help="Call the reverse generator in CycleGAN. Default: False")
    parser.add_argument("--diffusion_translator", type=DiffusionGenerator, help="Domain translation model to use in diffusion.", required=False, choices=list(DiffusionGenerator))
    args = parser.parse_args()
    save_path = args.save if args.save is not None else Path("examples")
    generate(args.model_type, args.ckpt_path, args.input_img, args.img_size, args.fran_input_age, save_path, args.fran_target_age, args.cyclegan_reverse, args.diffusion_translator)


if __name__ == "__main__":
    main()
