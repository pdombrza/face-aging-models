if __name__ == "__main__":
    import sys
    sys.path.append('../src')

import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os

from constants import CACD_SPLIT_DIR


def detect_faces(image_path):
    img = Image.open(image_path)
    cacd_image_size = 250
    detector = MTCNN(image_size=cacd_image_size, margin=0, keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
    boxes, _ = detector.detect(img)
    return boxes


def filter_multiple():
    img_dir = f"{CACD_SPLIT_DIR}/50Cent"

    for img in os.listdir(img_dir):
        faces = detect_faces(os.path.join(img_dir, img))
        if len(faces) > 1 or faces is None:
            print(img)


def main():
    filter_multiple()


if __name__ == "__main__":
    main()
