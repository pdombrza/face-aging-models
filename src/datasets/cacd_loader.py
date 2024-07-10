import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image
import pandas as pd


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
        self.young_images = self.metadata[(self.metadata['age'] >= 20 & self.metadata['age'] <= 30)]
        self.old_images = self.metadata[(self.metadata['age'] >= 50 & self.metadata['age'] <= 60)]

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

        return young_image, old_image



def main():
    meta_path = "cacd_meta/CACD_features_sex.csv"
    images_dir_path = "cacd_split"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    dataset = CACDDataset(meta_path, images_dir_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(next(iter(dataloader)))


if __name__ == "__main__":
    main()