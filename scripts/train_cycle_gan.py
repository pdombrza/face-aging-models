from datetime import timedelta
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.CycleGAN.cycle_gan_utils import CycleGANLossLambdaParams
from src.models.CycleGAN.train_cycle_gan import CycleGAN
from src.datasets.cacd_loader import CACDCycleGANDataset
from src.datasets.fgnet_loader import FGNETCycleGANDataset
from src.constants import FGNET_IMAGES_DIR, CACD_META_SEX_ANNOTATED_PATH, CACD_SPLIT_DIR


def train(
    dataset: str,
    loss_params: CycleGANLossLambdaParams | None = None,
    time_limit_s: int | None = None,
    n_valid_images: int = 16,
    epochs: int = 10,
    log_dir: Path | str = Path("../models/cycle_gan/tb_logs")
):
    if dataset not in ("cacd", "fgnet"):
        raise ValueError("Invalid dataset. Available: 'cacd', 'fgnet'.")
    if dataset == "fgnet":
        dataset = FGNETCycleGANDataset(FGNET_IMAGES_DIR)
    else:
        meta_path = CACD_META_SEX_ANNOTATED_PATH
        images_dir_path = CACD_SPLIT_DIR
        dataset = CACDCycleGANDataset(meta_path, images_dir_path)

    batch_size = 8
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    loss_params = loss_params if loss_params is not None else CycleGANLossLambdaParams
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)

    optimizer_params = {"lr": 0.0002, "betas": (0.5, 0.999)}
    model = CycleGAN(loss_params=loss_params, optimizer_params=optimizer_params)

    val_every_n_epochs = 1
    logger = TensorBoardLogger(log_dir, "cycle_gan")
    checkpoint_callback = ModelCheckpoint(
        filename="cycle_gan_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,
    )

    if time_limit_s is not None:
        time_limit = timedelta(seconds=time_limit_s)
    else:
        time_limit = timedelta(hours=24)

    trainer = L.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        max_time=time_limit,
        logger=logger
        )
    trainer.fit(model, train_loader, valid_loader)
    return model, trainer


def main():
    parser = ArgumentParser(description="Train CycleGAN model.")
    parser.add_argument("--dataset", help="Dataset to use for training. Possible options: 'cacd', 'fgnet'.", required=True)
    parser.add_argument("--maxtime", type=int, help="Time limit for training in seconds. Default: 86400.", required=False)
    parser.add_argument("--n_valid_images", type=int, help="Number of validation images. Default: 16", required=False, default=16)
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", required=False, default=10)
    parser.add_argument("--lambda_identity", type=float, help="Lambda param for identity Loss. Default: 5.0", required=False, default=5.0)
    parser.add_argument("--lambda_cycle", type=float, help="Lambda param for cycle Loss. Default: 10.0", required=False, default=10.0)
    parser.add_argument("--lambda_total", type=float, help="Lambda param for total Loss. Default: 0.5", required=False, default=0.5)
    parser.add_argument("--log_dir", type=Path, help="Path to save tensorboard log data.", required=False)
    parser.add_argument("--save", type=Path, help="Path to save trained model.", required=False)
    args = parser.parse_args()

    loss_params = CycleGANLossLambdaParams(args.lambda_identity, args.lambda_cycle, args.lambda_total)
    save_path = args.save if args.save is not None else Path("models/cycle_gan/")
    log_dir = args.log_dir if args.log_dir is not None else Path("models/cycle_gan/tb_logs")
    model, trainer = train(args.dataset, loss_params, args.maxtime, args.n_valid_images, args.epochs, log_dir)
    trainer.save_checkpoint(save_path)


if __name__ == "__main__":
    main()
