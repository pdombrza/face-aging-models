from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from PIL import Image
import numpy as np
import os
from src.constants import CACD_SPLIT_DIR


def get_face_embedding(img_path, detector, resnet):
    img = Image.open(img_path)
    img_cropped = detector(img)
    if img_cropped is not None:
        img_embedding = resnet(img_cropped.unsqueeze(0))
        return img_embedding.detach().numpy().flatten()
    return None


def detect():
    cacd_img_size = 250
    mtcnn = MTCNN(image_size=cacd_img_size, margin=0)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    img_dir = f"{CACD_SPLIT_DIR}/50Cent"
    embeddings = []
    img_paths = []
    for img in os.listdir(img_dir):
        embeddings.append(get_face_embedding(os.path.join(img_dir, img), mtcnn, resnet))
        img_paths.append(img)
    embeddings = np.vstack(embeddings)
    return embeddings, img_paths


def cluster(embeddings):
    clustering_model = DBSCAN(eps=0.6, min_samples=1, metric="euclidean")
    labels = clustering_model.fit_predict(embeddings)
    return labels


def print_labels():
    embeddings, imgs = detect()
    labels = cluster(embeddings)
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            print("Noise images:")
            label_indices = np.where(labels == label)[0]
            for index in label_indices:
                print(f"  {imgs[index]}")
        if label != -1:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > 1:
                print(f"Class {label} contains images:")
                for index in label_indices:
                    print(f"  {imgs[index]}")


def main():
    print_labels()


if __name__ == "__main__":
    main()
