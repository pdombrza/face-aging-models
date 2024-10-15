if __name__ == "__main__":
    import sys
    sys.path.append('src')

import os

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image

from models.FRAN.fran import Generator

CHECKPOINT_PATH = './models/fran/checkpoints/fran_epoch=04.ckpt'

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(CHECKPOINT_PATH)
    generator_weights = {k.replace("generator.", "", 1): v for k, v in model["state_dict"].items() if k.startswith("generator.")}
    generator_model = Generator()
    generator_model.load_state_dict(generator_weights)
    generator_model.to(device)
    target_ages = [13, 18, 23, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88]
    target_age = 88
    input_age = 28
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_image = 'data/interim/synthetic_images_full/seed0002.png_28.jpg'
    input_image_pil = Image.open(input_image)

    input_image_tensor = read_image(input_image, mode=ImageReadMode.RGB)
    input_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), input_age / 100)
    target_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), target_age / 100)

    input_tensor = torch.cat((input_image_tensor, input_age_embedding, target_age_embedding), dim=0)

    generator_model.eval()
    with torch.no_grad():
        output = generator_model(input_tensor.unsqueeze(0).to(device))

    np_test = np.array(input_image_pil)

    new_image = (output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255 + np.array(input_image_pil)).astype('uint8')

    sample_image = np.array(Image.fromarray(new_image).resize((np_test.shape[1], np_test.shape[0]))).astype('uint8')
    return sample_image




if __name__ == "__main__":
    main()