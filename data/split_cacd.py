import os
import shutil


NEWDIR = "cacd_split"
IMAGES_PATH = "CACD2000"

def create_directories(names):
    for name in names:
        os.makedirs(f"{NEWDIR}/{name}")


def main():
    images = os.listdir(IMAGES_PATH)
    names_set = {''.join(img.split('_')[1:-1]) for img in images}

    for img in images:
        name = ''.join(img.split('_')[1:-1])
        # print(img)
        shutil.copyfile(os.path.join(IMAGES_PATH, img), os.path.join(f"{NEWDIR}/{name}", img))

    # print(len(names_set))


if __name__ == "__main__":
    main()