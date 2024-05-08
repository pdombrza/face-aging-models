import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FGNETDataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.images = os.listdir(images_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]

        # load metadata
        img_meta = img_name.split('.')[0]
        person_id = img_meta[:3]
        person_age = int(img_meta[4:6])
        person_gender = 0 if img_meta[6] == 'M' else 1
        img_path = os.path.join(self.images_path, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return person_id, image, person_age, person_gender


def main():
    images_path = "FGNET/FGNET/individuals"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    dataset = FGNETDataset(images_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(next(iter(dataloader)))


if __name__ == "__main__":
    main()