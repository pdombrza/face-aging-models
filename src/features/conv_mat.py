import scipy.io as io
import numpy as np
import pandas as pd

from src.config import CACD_MAT_COLUMNS, CACD_MAT_PATH, CACD_MAT_STRUCTURE, CACD_CSV_PATH

def convert_to_dataframe(path, structure, columns=CACD_MAT_COLUMNS):
    mat = io.loadmat(path)

    features = mat[structure]
    features_dict = {col: feat.ravel() for col, feat in zip(columns, features[0][0])}
    del features_dict["feature"]

    features_dict["name"] = np.fromiter((el[0] for el in features_dict["name"]), dtype='<U40')

    df = pd.DataFrame.from_dict(data=features_dict)
    return df


if __name__ == "__main__":
    df = convert_to_dataframe(CACD_MAT_PATH, CACD_MAT_STRUCTURE)
    df.to_csv(CACD_CSV_PATH)