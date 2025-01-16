# This script will hopefully merge all 3 inference scripts together in the near future
from argparse import ArgumentParser
from pathlib import Path
import os
from enum import Enum

import torch
import torchvision.transforms as transforms

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
    FRAN = 'fran'
    DIFFUSION = 'diffusion'

    def __str__(self):
        return self.value

def main():
    parser = ArgumentParser(description="Generate images from model.")
    parser.add_argument("--model_type", type=ModelType, help="Model type", required=True, choices=list(ModelType))
    parser.add_argument("--ckpt_path", type=Path, help="Path of model checkpoint to use.", required=True)
    parser.add_argument("--input_img", type=Path, help="Input image path.", required=True)
    parser.add_argument("--input_age", type=Path, help="Input image age.", required=True)
    parser.add_argument("--input_gender", type=Gender, help="Input image gender", choices=list(Gender), required=True)
    parser.add_argument("--save", type=Path, help="Path to save image.", required=False)
    args = parser.parse_args()



if __name__ == "__main__":
    main()