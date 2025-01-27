#!/bin/bash

declare -A DATASETS
DATASETS=(
    ["synthetic"]="https://www.kaggle.com/api/v1/datasets/download/pdombrza/synthetic-aged-images"
    ["cacd"]="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    ["fgnet"]="https://www.image-net.org/archive/imagenet_data.tar.gz"
)

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Available datasets:"
    for key in "${!DATASETS[@]}"; do
        echo "  - $key"
    done
    exit 1
fi

DATASET_NAME=$1

if [[ ! -v DATASETS[$DATASET_NAME] ]]; then
    echo "Error: Dataset '$DATASET_NAME' not found."
    echo "Available datasets:"
    for key in "${!DATASETS[@]}"; do
        echo "  - $key"
    done
    exit 1
fi

if [ DATASET_NAME == "synthetic" ]; then
    curl -L -o data/synthetic-aged-images.zip ${DATASETS[$DATASET_NAME]}
    unzip -q data/synthetic-aged-images.zip -d data/processed/
    mv data/processed/out data/processed/synthetic_images_full
    exit 0
else
    curl -L -o data/${DATASET_NAME}.tar.gz ${DATASETS[$DATASET_NAME]}
    tar -xzf data/${DATASET_NAME}.tar.gz -C data/processed/
fi