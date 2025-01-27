from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image

from src.models.diffusion.diffusion import DiffusionModel
from src.models.FRAN.fran import Generator

CHECKPOINT_PATH = "models/diffusion_test3/diffusion_epoch=12.ckpt"

def main():
    model = DiffusionModel.load_from_checkpoint(CHECKPOINT_PATH, generator=Generator(in_channels=3))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    input_image = "data/processed/cacd_split/AaronPaul/25_Aaron_Paul_0001.jpg"
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_image_tensor = transform(read_image(input_image, mode=ImageReadMode.RGB))

    model.eval()
    with torch.no_grad():
        output = model(input_image_tensor.unsqueeze(0).to(device), release_time=1)

    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    min_val, max_val = output.min(), output.max()
    print(min_val, max_val)
    new_image_normalized = 255 * (output - min_val) / (max_val - min_val)
    new_image_normalized = new_image_normalized.astype('uint8')
    sample_image = Image.fromarray(new_image_normalized)
    sample_image.save("examples/output_image_diffusion.png")

if __name__ == "__main__":
    main()
