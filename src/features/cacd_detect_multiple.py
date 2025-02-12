from facenet_pytorch import MTCNN
import torch
from PIL import Image
import os
from tqdm import tqdm


def detect_faces(image_path):
    img = Image.open(image_path)
    cacd_image_size = 250
    detector = MTCNN(
        image_size=cacd_image_size,
        margin=0,
        keep_all=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    boxes, _ = detector.detect(img)
    return boxes


def filter_multiple(remove=False):
    img_directories = "../../data/processed/cacd_split"
    count = 0
    for img_dir in tqdm(os.listdir(img_directories)):
        img_dir_path = os.path.join(img_directories, img_dir)
        for img in os.listdir(img_dir_path):
            img_path = os.path.join(img_dir_path, img)
            try:
                faces = detect_faces(img_path)
            except OSError as e:
                print(f"OSError: {img}, {e}")
                count += 1
                if remove:
                    os.remove(img_path)
                continue
            if faces is None:
                print(f"None: {img}")
                count += 1
                if remove:
                    os.remove(img_path)
            elif len(faces) > 1:
                print(img)
                count += 1
                if remove:
                    os.remove(img_path)
    print(count)


def main():
    filter_multiple(True)


if __name__ == "__main__":
    main()
