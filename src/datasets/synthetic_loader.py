import os
from itertools import permutations
import torch
from kornia.augmentation import AugmentationSequential
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image


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
            self.transform = transforms.Compose(
                [
                    transforms.ConvertImageDtype(dtype=torch.float),
                    transforms.Resize((160, 160)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        image_pair = self.image_pairs[index]
        input_image = read_image(
            os.path.join(self.images_path, image_pair[0]), mode=ImageReadMode.RGB
        )
        target_image = read_image(
            os.path.join(self.images_path, image_pair[1]), mode=ImageReadMode.RGB
        )

        if isinstance(self.transform, AugmentationSequential) is True:
            input_image = self.transform(input_image).squeeze()
            target_image = self.transform(
                target_image, params=self.transform._params
            ).squeeze()
        else:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        input_age = int(image_pair[0][13:15])
        target_age = int(image_pair[1][13:15])
        age_tensor_input = torch.full(
            (1, input_image.shape[1], input_image.shape[2]), input_age / 100
        )  # 1 x W x H
        age_tensor_target = torch.full(
            (1, target_image.shape[1], target_image.shape[2]), target_age / 100
        )  # 1 x W x H

        tensor_input = torch.cat(
            (input_image, age_tensor_input, age_tensor_target), dim=0
        )

        return {
            "input": tensor_input,
            "input_img": input_image,
            "target_img": target_image,
            "target_age": age_tensor_target,
        }
