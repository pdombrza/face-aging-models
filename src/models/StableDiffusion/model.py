import torch
import numpy as np
from tqdm import tqdm
from scheduler import DDPM

IMG_WIDTH = 512
IMG_HEIGHT = 512
VAE_LATENT_WIDTH = IMG_WIDTH // 8
VAE_LATENT_HEIGHT = IMG_HEIGHT // 8


def generate(
        prompt: str,
        uncond_prompt: str,
        input_image=None,
        strength=0.8,
        classifier_free_guidance=True,
        cfg_scale=7.5, sampler="ddpm",
        pretrained_models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
        ):
    ...
