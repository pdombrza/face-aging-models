from datetime import timedelta
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.FRAN.fran_utils import FRANLossLambdaParams
from src.models.FRAN.train_fran import FRAN
from src.datasets.cacd_loader import CACDFRANDataset
from src.datasets.fgnet_loader import FGNETFRANDataset
from src.constants import FGNET_IMAGES_DIR, CACD_META_SEX_ANNOTATED_PATH, CACD_SPLIT_DIR


def train(
    dataset: str,
    loss_params: FRANLossLambdaParams | None = None,
    time_limit_s: int | None = None,
    n_valid_images: int = 16,
    epochs: int = 10,
    log_dir: Path | str = Path("models/fran/tb_logs")
):
    if dataset not in ("cacd", "fgnet"):
        raise ValueError("Invalid dataset. Available: 'cacd', 'fgnet'.")
    if dataset == "fgnet":
        dataset = FGNETFRANDataset(FGNET_IMAGES_DIR)
    else:
        meta_path = CACD_META_SEX_ANNOTATED_PATH
        images_dir_path = CACD_SPLIT_DIR
        dataset = CACDFRANDataset(meta_path, images_dir_path)

    batch_size = 8
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, (train_size, n_valid_images))
    loss_params = loss_params if loss_params is not None else FRANLossLambdaParams
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=n_valid_images, shuffle=False, num_workers=8, pin_memory=True)
    model = FRAN(loss_params=loss_params)

    val_every_n_epochs = 1
    logger = TensorBoardLogger(log_dir, "fran")
    checkpoint_callback = ModelCheckpoint(
        filename="fran_{epoch:02d}",
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
    parser = ArgumentParser(description="Train FRAN model.")
    parser.add_argument("--dataset", help="Dataset to use for training. Possible options: 'cacd', 'fgnet'.", required=True)
    parser.add_argument("--maxtime", type=int, help="Time limit for training in seconds. Default: 86400.", required=False)
    parser.add_argument("--n_valid_images", type=int, help="Number of validation images. Default: 16", required=False, default=16)
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", required=False, default=10)
    parser.add_argument("--lambda_l1", type=float, help="Lambda param for L1 Loss. Default: 1.0", required=False, default=1.0)
    parser.add_argument("--lambda_adv", type=float, help="Lambda param for adversarial Loss. Default: 0.05", required=False, default=0.05)
    parser.add_argument("--lambda_lpips", type=float, help="Lambda param for LPIPS Loss. Default: 1.0", required=False, default=1.0)
    parser.add_argument("--log_dir", type=Path, help="Path to save tensorboard log data.", required=False)
    parser.add_argument("--save", type=Path, help="Path to save trained model.", required=False)
    args = parser.parse_args()

    loss_params = FRANLossLambdaParams(args.lambda_l1, args.lambda_lpips, args.lambda_adv)
    save_path = args.save if args.save is not None else Path("models/fran/")
    log_dir = args.log_dir if args.log_dir is not None else Path("models/fran/tb_logs")
    model, trainer = train(args.dataset, loss_params, args.maxtime, args.n_valid_images, args.epochs, log_dir)
    trainer.save_checkpoint(save_path)


if __name__ == "__main__":
    main()
