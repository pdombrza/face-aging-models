if __name__ == "__main__":
    import sys
    sys.path.append('src')

import os

from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image
from generate import imsave

from models.FRAN.fran import Generator

CHECKPOINT_PATH = './models/fran/checkpoints/fran_epoch=04.ckpt'


def main():
    os.makedirs("examples", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(CHECKPOINT_PATH)
    generator_weights = {k.replace("generator.", "", 1): v for k, v in model["state_dict"].items() if k.startswith("generator.")}
    generator_model = Generator()
    generator_model.load_state_dict(generator_weights)
    generator_model.to(device)
    target_ages = [13, 18, 23, 33, 38, 48, 53, 58, 63, 68, 73, 78]
    target_age = 83
    input_age = 43
    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((160, 160)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_image = f'data/interim/synthetic_images_full/seed0002.png_{input_age}.jpg'
    #input_image = "ryan_gosling.jpeg"
    input_image_pil = Image.open(input_image).resize((160, 160))
    input_image_pil.save(f"sample_image28.png", "PNG")

    input_image_tensor = transform(read_image(input_image, mode=ImageReadMode.RGB))
    input_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), input_age / 100)

    out_image_array = []
    for target_age in target_ages:
        target_age_embedding = torch.full((1, input_image_tensor.shape[1], input_image_tensor.shape[2]), target_age / 100)
        input_tensor = torch.cat((input_image_tensor, input_age_embedding, target_age_embedding), dim=0)
        generator_model.eval()
        with torch.no_grad():
            output = generator_model(input_tensor.unsqueeze(0).to(device))

        # print(output)
        np_test = np.array(input_image_pil)

        #new_image_out = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        new_image_out = output.squeeze(0).cpu()
        min_val, max_val = new_image_out.min(), new_image_out.max()
        new_image_normalized = 255 * (new_image_out - min_val) / (max_val - min_val)
        # new_image_normalized = new_image_normalized.astype('uint8')
        out_image_array.append(new_image_normalized)
        # print(new_image_normalized.shape)

        # new_image = (new_image_normalized + np.array(input_image_pil)).astype('uint8')
        # sample_image = Image.fromarray(new_image_normalized)
        # sample_image.save(f"examples/output_image{target_age}.png", "PNG")
    imsave(torchvision.utils.make_grid(out_image_array, nrow=len(out_image_array) // 3))

    return


if __name__ == "__main__":
    main()