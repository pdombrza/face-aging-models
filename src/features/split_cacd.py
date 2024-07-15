import os
import shutil

from src.constants import CACD_IMAGES_PATH, CACD_SPLIT_DIR


def create_directories(names):
    for name in names:
        os.makedirs(f"{CACD_SPLIT_DIR}/{name}")


def main():
    images = os.listdir(CACD_IMAGES_PATH)
    names_set = {''.join(img.split('_')[1:-1]) for img in images}

    for img in images:
        name = ''.join(img.split('_')[1:-1])
        # print(img)
        shutil.copyfile(os.path.join(CACD_IMAGES_PATH, img), os.path.join(f"{CACD_SPLIT_DIR}/{name}", img))

    # print(len(names_set))


if __name__ == "__main__":
    main()