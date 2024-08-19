if __name__ == "__main__":
    import sys
    sys.path.append('../src')

import os
from itertools import permutations
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image
from constants import FGNET_IMAGES_DIR


def gen_fgnet_img_pairs_fran(images_path):
    images = os.listdir(images_path)
    image_path_pairs = []
    images_by_id = {i: [] for i in range(1, 83)}
    for img in images:
        img_meta = img.split(".")[0]
        person_id = int(img_meta[:3])
        images_by_id[person_id].append(img)

    for person in images_by_id:
        for perm in permutations(images_by_id[person], 2):
            image_path_pairs.append(perm)

    return image_path_pairs


class FGNETDataset(Dataset):
    def __init__(self, images_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.images = os.listdir(images_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]

        # load metadata
        img_meta = img_name.split(".")[0]
        person_id = img_meta[:3]
        person_age = int(img_meta[4:6])
        person_gender = 0 if img_meta[6] == "M" else 1
        img_path = os.path.join(self.images_path, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return person_id, image, person_age, person_gender


class FGNETCycleGANDataset(Dataset):
    def __init__(self, images_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.transform = transform
        self.images = os.listdir(images_path)
        self.young_images = list(
            filter(lambda x: int(x[4:6]) < 20 and int(x[4:6]) > 10, self.images)
        )
        self.old_images = list(filter(lambda x: int(x[4:6]) > 40, self.images))
        # min_length = len(self)
        # self.young_images = self.young_images[:min_length]
        # self.old_images = self.old_images[:min_length]

    def __len__(self):
        return min(len(self.young_images), len(self.old_images))

    def __getitem__(self, index):
        young_image = read_image(
            os.path.join(self.images_path, self.young_images[index]),
            mode=ImageReadMode.RGB,
        )
        old_image = read_image(
            os.path.join(self.images_path, self.old_images[index]),
            mode=ImageReadMode.RGB,
        )

        if self.transform is not None:
            young_image = self.transform(young_image)
            old_image = self.transform(old_image)

        return {
            "young_image": young_image,
            "old_image": old_image,
        }


class FGNETFRANDataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.image_pairs = gen_fgnet_img_pairs_fran(images_path)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        image_pair = self.image_pairs[index]
        input_image = read_image(os.path.join(self.images_path, image_pair[0]), mode=ImageReadMode.RGB)
        target_image = read_image(os.path.join(self.images_path, image_pair[1]), mode=ImageReadMode.RGB)

        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        input_age = int(image_pair[0][4:6])
        target_age = int(image_pair[1][4:6])
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
    images_path = FGNET_IMAGES_DIR
    transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(dtype=torch.float),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = FGNETFRANDataset(images_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    res = next(iter(dataloader))
    print(torch.min(res["input"]))
    # image_pairs = gen_fgnet_img_pairs_fran(images_path)
    # print(image_pairs)


if __name__ == "__main__":
    main()
