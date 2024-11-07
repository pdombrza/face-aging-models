if __name__ == "__main__":
    import sys
    sys.path.append('../src')

import os
from itertools import permutations
from PIL import Image
import torch
from kornia.augmentation import AugmentationSequential
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image
from constants import SYNTHETIC_IMAGES_FULL

TARGET_AGES = [13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83]

def gen_synthetic_img_pairs_fran(images_path):
    images = os.listdir(images_path)
    image_path_pairs = []
    images_by_id = {int(img[4:8]): [] for img in images}
    for img in images:
        if len(img) > 13:
            person_id = int(img[4:8])
            images_by_id[person_id].append(img)

    for person in images_by_id:
        for perm in permutations(images_by_id[person], 2):
            image_path_pairs.append(perm)

    return image_path_pairs


class SynthFRANDataset(Dataset):
    def __init__(self, images_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.image_pairs = gen_synthetic_img_pairs_fran(images_path)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ConvertImageDtype(dtype=torch.float),
                transforms.Resize((160, 160)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        image_pair = self.image_pairs[index]
        input_image = read_image(os.path.join(self.images_path, image_pair[0]), mode=ImageReadMode.RGB)
        target_image = read_image(os.path.join(self.images_path, image_pair[1]), mode=ImageReadMode.RGB)


        if isinstance(self.transform, AugmentationSequential) is True:
            input_image = self.transform(input_image).squeeze()
            target_image = self.transform(target_image, params=self.transform._params).squeeze()
            print("kornia")
        else:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        input_age = int(image_pair[0][13:15])
        target_age = int(image_pair[1][13:15])
        age_tensor_input = torch.full((1, input_image.shape[1], input_image.shape[2]), input_age / 100)  # 1 x W x H
        age_tensor_target = torch.full((1, target_image.shape[1], target_image.shape[2]), target_age / 100)  # 1 x W x H

        tensor_input = torch.cat((input_image, age_tensor_input, age_tensor_target), dim=0)

        return {
            "input": tensor_input,
            "input_img": input_image,
            "target_img": target_image,
            "target_age": age_tensor_target,
        }


def main():
    images_path = SYNTHETIC_IMAGES_FULL
    import kornia
    from kornia.augmentation import AugmentationSequential
    transform = AugmentationSequential(
        transforms.ConvertImageDtype(dtype=torch.float),
        kornia.augmentation.RandomCrop((256, 256)),
        kornia.augmentation.ColorJitter(p=0.5),
        kornia.augmentation.RandomBoxBlur(p=0.1),
        kornia.augmentation.RandomBrightness(p=0.5),
        kornia.augmentation.RandomContrast(p=0.5),
        kornia.augmentation.RandomAffine(degrees=(-30, 30), scale=(0.5, 1.5), p=0.6),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        same_on_batch=False,
    )

    dataset = SynthFRANDataset(images_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # res = next(iter(dataloader))
    # print(res)
    # print(torch.min(res["input"]))
    # image_pairs = gen_synthetic_img_pairs_fran(images_path)
    # for image in image_pairs:
    #     if image[0][4:8] == '1868':
    #         print(image)


if __name__ == "__main__":
    main()