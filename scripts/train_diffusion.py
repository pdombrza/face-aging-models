from datetime import timedelta
from argparse import ArgumentParser
from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint

from src.models.diffusion.diffusion import DiffusionModel
from src.models.FRAN.fran import Generator
from src.datasets.cacd_loader import CACDCycleGANDataset
from src.datasets.fgnet_loader import FGNETCycleGANDataset
from src.constants import FGNET_IMAGES_DIR, CACD_META_SEX_ANNOTATED_PATH, CACD_SPLIT_DIR

torch.set_float32_matmul_precision('high')

def train(
    dataset: str,
    generator: nn.Module | None = None,
    age_type: int = 1,
    gender_type: int = 0,
    lambda_cycle: float = 10.0,
    time_limit_s: int | None = None,
    n_valid_images: int = 16,
    epochs: int = 10,
    batch_size: int = 8,
    img_size: int = 244,
    ckpt_save_dir: Path | str = Path("models/diffusion"),
    log_dir: Path | str = Path("models/diffusion/tb_logs"),
    ckpt_load_path: Path | str | None = None,
):

    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((img_size, img_size)) if img_size != 250 else transforms.Lambda(lambda x: x),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset == "fgnet":
        dataset = FGNETCycleGANDataset(FGNET_IMAGES_DIR, transform)
    elif dataset == "cacd":
        meta_path = CACD_META_SEX_ANNOTATED_PATH
        images_dir_path = CACD_SPLIT_DIR
        dataset = CACDCycleGANDataset(meta_path, images_dir_path, age_type, gender_type, transform)
    else:
        raise ValueError("Invalid dataset. Available: 'cacd', 'fgnet'.")

    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)

    if generator == 'cyclegan':
        model = DiffusionModel(lambda_cycle=lambda_cycle)
    elif generator == 'unet':
        model = DiffusionModel(generator=Generator(in_channels=3), lambda_cycle=lambda_cycle)

    val_every_n_epochs = 1
    logger = TensorBoardLogger(log_dir, "diffusion")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_save_dir,
        filename="diffusion_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,
    )

    exception_callback = OnExceptionCheckpoint(
        dirpath=ckpt_save_dir,
        filename="diffusion_{epoch}-{step}_ex",
    )

    if time_limit_s is not None:
        time_limit = timedelta(seconds=time_limit_s)
    else:
        time_limit = timedelta(hours=24)

    trainer = L.Trainer(
        callbacks=[checkpoint_callback, exception_callback],
        max_epochs=epochs,
        max_time=time_limit,
        logger=logger
        )
    trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt_load_path)
    return model, trainer


def main():
    parser = ArgumentParser(description="Train diffusion model.")
    parser.add_argument("--dataset", help="Dataset to use for training", choices=["fgnet", "cacd"], required=True)
    parser.add_argument("--generator", help="Generator for translation in training.", required=True, choices=["unet", "cyclegan"], default='cyclegan')
    parser.add_argument("--age_type", type=int, help="Available age transformation intervals. 1 - 20-30->50-60, 2 - 20-30->35-45, 3 - 35-45-> 50-60. Default: 1", required=False, default=1),
    parser.add_argument("--gender_type", type=int, help="Wether to train on a gendered dataset or not. 0 - full dataset, 1 - male only, 2 - female only. Default: 0", choices=[0, 1, 2], required=False, default=0),
    parser.add_argument("--maxtime", type=int, help="Time limit for training in seconds. Default: 86400.", required=False)
    parser.add_argument("--n_valid_images", type=int, help="Number of validation images. Default: 16", required=False, default=16)
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", required=False, default=10)
    parser.add_argument("--batch", type=int, help="Batch size. Default: 8", required=False, default=8)
    parser.add_argument("--img_size", type=int, help="Training image size. Default: 244", required=False, default=244)
    parser.add_argument("--lambda_cycle", type=float, help="Lambda param for cycle Loss. Default: 10.0", required=False, default=10.0)
    parser.add_argument("--log_dir", type=Path, help="Path to save tensorboard log data.", required=False)
    parser.add_argument("--ckpt_load", type=Path, help="Path to load checkpoint. ", required=False)
    parser.add_argument("--save", type=Path, help="Path to save trained model.", required=False)
    args = parser.parse_args()

    lambda_cycle = args.lambda_cycle
    save_path = args.save if args.save is not None else Path("models/diffusion/")
    log_dir = args.log_dir if args.log_dir is not None else Path("models/diffusion/tb_logs")
    img_size = max(min(args.img_size, 244), 16)
    model, trainer = train(args.dataset, args.generator, args.age_type, args.gender_type, lambda_cycle, args.maxtime, args.n_valid_images, args.epochs, args.batch, img_size, save_path, log_dir, args.ckpt_load)
    trainer.save_checkpoint(os.path.join(save_path, f"diffusion_fin"))


if __name__ == "__main__":
    main()
