from datetime import timedelta
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import kornia
from kornia.augmentation import AugmentationSequential
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint

from src.models.FRAN.fran_utils import FRANLossLambdaParams
from src.models.FRAN.train_fran import FRAN
from src.datasets.synthetic_loader import SynthFRANDataset
from src.constants import SYNTHETIC_IMAGES_FULL

torch.set_float32_matmul_precision('high')

def train(
    dataset: str,
    loss_params: FRANLossLambdaParams | None = None,
    time_limit_s: int | None = None,
    n_valid_images: int = 16,
    epochs: int = 10,
    batch_size: int = 8,
    img_size: int = 512,
    ckpt_save_dir: Path | str = Path("models/fran"),
    log_dir: Path | str = Path("models/fran/tb_logs"),
    ckpt_load_path: Path | str | None = None,
):

    transform = AugmentationSequential(
        transforms.ConvertImageDtype(dtype=torch.float),
        kornia.augmentation.RandomCrop((img_size // 2, img_size // 2)),
        kornia.augmentation.ColorJitter(p=0.5),
        kornia.augmentation.RandomBoxBlur(p=0.1),
        kornia.augmentation.RandomBrightness(p=0.5),
        kornia.augmentation.RandomContrast(p=0.5),
        kornia.augmentation.RandomAffine(degrees=(-30, 30), scale=(0.5, 1.5), p=0.6),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        same_on_batch=False,
    )

    if dataset not in ("synthetic"):
        raise ValueError("Invalid dataset. Available: 'synthetic'.")
    if dataset == "synthetic":
        dataset = SynthFRANDataset(SYNTHETIC_IMAGES_FULL, transform)

    batch_size = batch_size
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    loss_params = loss_params if loss_params is not None else FRANLossLambdaParams
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)
    model = FRAN(loss_params=loss_params)

    val_every_n_epochs = 1
    logger = TensorBoardLogger(log_dir, "fran")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_save_dir,
        filename="fran_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,
    )

    exception_callback = OnExceptionCheckpoint(
        dirpath=ckpt_save_dir,
        filename="fran_{epoch}-{step}_ex",
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
    parser = ArgumentParser(description="Train FRAN model.")
    parser.add_argument("--dataset", help="Dataset to use for training. Possible options: 'synthetic'.", required=True)
    parser.add_argument("--maxtime", type=int, help="Time limit for training in seconds. Default: 86400.", required=False)
    parser.add_argument("--n_valid_images", type=int, help="Number of validation images. Default: 16", required=False, default=16)
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", required=False, default=10)
    parser.add_argument("--batch", type=int, help="Batch size. Default: 8", required=False, default=8)
    parser.add_argument("--img_size", type=int, help="Training image size. Default: 244", required=False, default=244)
    parser.add_argument("--lambda_l1", type=float, help="Lambda param for L1 Loss. Default: 1.0", required=False, default=1.0)
    parser.add_argument("--lambda_adv", type=float, help="Lambda param for adversarial Loss. Default: 0.05", required=False, default=0.05)
    parser.add_argument("--lambda_lpips", type=float, help="Lambda param for LPIPS Loss. Default: 1.0", required=False, default=1.0)
    parser.add_argument("--log_dir", type=Path, help="Path to save tensorboard log data.", required=False)
    parser.add_argument("--ckpt_load", type=Path, help="Path to load checkpoint. ", required=False)
    parser.add_argument("--save", type=Path, help="Path to save trained model.", required=False)
    args = parser.parse_args()

    loss_params = FRANLossLambdaParams(args.lambda_l1, args.lambda_lpips, args.lambda_adv)
    save_path = args.save if args.save is not None else Path("models/fran/")
    log_dir = args.log_dir if args.log_dir is not None else Path("models/fran/tb_logs")
    model, trainer = train(args.dataset, loss_params, args.maxtime, args.n_valid_images, args.epochs, args.batch, args.img_size, log_dir, args.ckpt_load)
    trainer.save_checkpoint(save_path)


if __name__ == "__main__":
    main()
