import scipy.io as io
import numpy as np
import pandas as pd


PATH = "celebrity2000_meta.mat"
STRUCTURE = "celebrityImageData"
COLUMNS = ("age", "identity", "year", "feature", "rank", "lfw", "birth", "name")
CSV_PATH = "CACD_features.csv"


def convert_to_dataframe(path, structure, columns=COLUMNS):
    mat = io.loadmat(path)

    features = mat[structure]
    features_dict = {col: feat.ravel() for col, feat in zip(columns, features[0][0])}
    del features_dict["feature"]

    features_dict["name"] = np.fromiter((el[0] for el in features_dict["name"]), dtype='<U40')

    df = pd.DataFrame.from_dict(data=features_dict)
    return df


if __name__ == "__main__":
    df = convert_to_dataframe(PATH, STRUCTURE)
    df.to_csv(CSV_PATH)