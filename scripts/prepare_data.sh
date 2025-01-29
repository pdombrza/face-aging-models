#!/bin/bash

declare -A DATASETS
DATASETS=(
    ["synthetic"]="https://www.kaggle.com/api/v1/datasets/download/pdombrza/synthetic-aged-images"
    ["cacd"]="https://www.kaggle.com/api/v1/datasets/download/pdombrza/cacd-filtered-dataset"
    ["fgnet"]="https://www.kaggle.com/api/v1/datasets/download/pdombrza/fgnet-gendered"
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

if [ DATASET_NAME == "synthetic" ]; then
    curl -L -o data/synthetic-aged-images.zip ${DATASETS[$DATASET_NAME]}
    unzip -q data/synthetic-aged-images.zip -d data/processed/
    mv data/processed/out data/processed/synthetic_images_full
    exit 0
elif [ DATASET_NAME == "fgnet" ]; then
    curl -L -o data/fgnet.zip ${DATASETS[$DATASET_NAME]}
    unzip -q data/synthetic-aged-images.zip -d data/processed/
elif [ DATASET_NAME == "cacd" ]; then
    curl -L -o data/cacd-split.zip ${DATASETS[$DATASET_NAME]}
    unzip -q data/cacd-split.zip -d data/processed/
    mkdir -p data/processed/cacd_meta
    mv data/processed/CACD_features_sex.csv data/processed/cacd_meta
else
    echo "Error: Dataset '$DATASET_NAME' not found."
    echo "Available datasets:"
    for key in "${!DATASETS[@]}"; do
        echo "  - $key"
    done
    exit 1
fi