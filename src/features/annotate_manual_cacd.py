import pandas as pd
import numpy as np
from copy import deepcopy

from src.constants import (
    CACD_META_SEX_ANNOTATED_PATH,
    CACD_MANUAL_READ,
    CACD_META_SEX_ANNOTATED_PATH,
)


def main():
    save = True

    annotated = pd.read_csv(CACD_MANUAL_READ)
    cacd_features = deepcopy(pd.read_csv(CACD_META_SEX_ANNOTATED_PATH))
    cacd_features["name_only"] = cacd_features["name"].map(lambda x: x.split("_")[1])

    merged_df = pd.merge(
        cacd_features,
        annotated,
        left_on=cacd_features["name_only"],
        right_on=annotated["name"],
        how="left",
    )
    merged_df["gender"] = merged_df["gender_x"].fillna("") + merged_df[
        "gender_y"
    ].fillna("")
    merged_df = merged_df.drop(
        ["name_only", "key_0", "Unnamed: 0", "name_y", "gender_x", "gender_y"], axis=1
    )
    merged_df.rename(columns={"name_x": "name"}, inplace=True)
    merged_df["gender"] = merged_df["gender"].replace("", np.nan)

    nan_count = merged_df["gender"].isna().sum()
    nan_rows = merged_df[merged_df["gender"].isna()]
    print(merged_df.head())
    # print(merged_df)

    if save:
        pd.DataFrame.to_csv(merged_df, CACD_META_SEX_ANNOTATED_PATH)


if __name__ == "__main__":
    main()
