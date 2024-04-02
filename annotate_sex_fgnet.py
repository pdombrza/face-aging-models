import os
import shutil


IMAGES_DIR = "./images/"
INDIVIDUALS_DIR = "./individuals"


def main():
    image_names = list(os.listdir(IMAGES_DIR))
    individuals = {name[:3] for name in image_names}
    ind_files = []
    for i in individuals:
        for iname in image_names:
            if i in iname:
                ind_files.append(iname)
                break

    for img in ind_files:
        shutil.copyfile(
            os.path.join(IMAGES_DIR, img), os.path.join(INDIVIDUALS_DIR, img)
        )


if __name__ == "__main__":
    main()
