from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image

from src.models.CycleGAN.train_cycle_gan import CycleGAN

CHECKPOINT_PATH = "models/cycle_gan/cycle_gan_female_aug/cycle_gan_fin"

def main():
    model = CycleGAN.load_from_checkpoint(CHECKPOINT_PATH, optimizer_params={"lr": 0.0002, "betas": (0.5, 0.999)})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    input_image = "data/interim/cacd_split/AlexandraDaddario/18_Alexandra_Daddario_0010.jpg"
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((244, 244)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_image_tensor = transform(read_image(input_image, mode=ImageReadMode.RGB))

    model.eval()
    with torch.no_grad():
        output = model(input_image_tensor.unsqueeze(0).to(device))

    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    min_val, max_val = output.min(), output.max()
    new_image_normalized = 255 * (output - min_val) / (max_val - min_val)
    new_image_normalized = new_image_normalized.astype('uint8')
    sample_image = Image.fromarray(new_image_normalized)
    sample_image.save("examples/output_image_cycle_female.png")

if __name__ == "__main__":
    main()
