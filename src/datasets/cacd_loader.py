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
import pandas as pd
from constants import CACD_META_SEX_ANNOTATED_PATH, CACD_SPLIT_DIR


def gen_cacd_img_pairs_fran(meta_df: pd.DataFrame) -> list[tuple]:
    images_by_id = meta_df.groupby('identity')['name'].apply(list).to_dict()
    image_path_pairs = []

    for identity, images in images_by_id.items():
        for pair in permutations(images, 2):
            if pair[0][:2] != pair[1][:2]:
                image_path_pairs.append(pair)

    return image_path_pairs


class CACDDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        img_name = self.metadata.iloc[index]['name']

        # get metadata
        person_age = int(self.metadata.iloc[index]['age'])
        person_gender = 0 if self.metadata.iloc[index]['gender'] == 'M' else 1
        celeb_name = ''.join(img_name.split('_')[1:-1])

        # the image itself
        dir_path = os.path.join(self.img_root_dir, celeb_name)
        image = Image.open(os.path.join(dir_path, img_name)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return celeb_name, image, person_age, person_gender


class CACDCycleGANDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.img_root_dir = img_root_dir
        self.young_images = self.metadata[(self.metadata['age'] >= 20) & (self.metadata['age'] <= 30)]
        self.old_images = self.metadata[(self.metadata['age'] >= 50) & (self.metadata['age'] <= 60)]

    def __len__(self):
        return min(len(self.young_images), len(self.old_images))

    def __getitem__(self, index):
        young_img_name = self.young_images.iloc[index]['name']
        young_celeb_name = ''.join(young_img_name.split('_')[1:-1])
        young_dir_path = os.path.join(self.img_root_dir, young_celeb_name)
        young_image = read_image(os.path.join(young_dir_path, young_img_name), mode=ImageReadMode.RGB)

        old_img_name = self.old_images.iloc[index]['name']
        old_celeb_name = ''.join(old_img_name.split('_')[1:-1])
        old_dir_path = os.path.join(self.img_root_dir, old_celeb_name)
        old_image = read_image(os.path.join(old_dir_path, old_img_name), mode=ImageReadMode.RGB)

        if self.transform is not None:
            young_image = self.transform(young_image)
            old_image = self.transform(old_image)

        return {
            "young_image": young_image,
            "old_image": old_image,
        }


class CACDFRANDataset(Dataset):
    # Maybe merge with FGNET dataset?
    def __init__(self, csv_file, img_root_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.img_root_dir = img_root_dir
        self.image_pairs = gen_cacd_img_pairs_fran(self.metadata)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        image_pair = self.image_pairs[index]
        celeb_dir_name = ''.join(image_pair[0].split('_')[1:-1])
        celeb_dir_path = os.path.join(self.img_root_dir, celeb_dir_name)
        input_image = read_image(os.path.join(celeb_dir_path, image_pair[0]), mode=ImageReadMode.RGB)
        target_image = read_image(os.path.join(celeb_dir_path, image_pair[1]), mode=ImageReadMode.RGB)

        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        input_age = int(image_pair[0][:2])
        target_age = int(image_pair[1][:2])
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
    meta_path = CACD_META_SEX_ANNOTATED_PATH
    images_dir_path = CACD_SPLIT_DIR
    transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(dtype=torch.float),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = CACDFRANDataset(meta_path, images_dir_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(next(iter(dataloader)))
    # meta_df = pd.read_csv(meta_path)
    # print(gen_cacd_img_pairs_fran(meta_df))


if __name__ == "__main__":
    main()