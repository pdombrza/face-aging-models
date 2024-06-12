import pandas as pd
import numpy as np
from copy import deepcopy
import csv

CACD_PATH = "cacd_meta/CACD_features.csv"
MANUAL_SAVE = "cacd_meta/manual_save.csv"
MANUAL_READ = "cacd_meta/manual.csv"
MALE_NAMES = "cacd_meta/male.csv"
FEMALE_NAMES = "cacd_meta/female.csv"

def write_names_to_csv(data, path):
    with open(path, "w") as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        for name in data:
            writer.writerow([name+','])


def main():
    save = False

    cacd_features = deepcopy(pd.read_csv(CACD_PATH))
    cacd_features['name_only'] = cacd_features['name'].map(lambda x: x.split("_")[1])
    male_names = pd.read_csv(MALE_NAMES)
    female_names = pd.read_csv(FEMALE_NAMES)
    male_col = pd.DataFrame.from_dict({"gender": ['M' for _ in range(len(male_names))]})
    female_col = pd.DataFrame.from_dict({"gender": ['F' for _ in range(len(female_names))]})
    male_names = pd.concat((male_names, male_col), axis=1)
    female_names = pd.concat((female_names, female_col), axis=1)

    merged_df = pd.merge(cacd_features, male_names, left_on=cacd_features['name_only'], right_on=male_names['name'], how='left')

    female_names = pd.merge(female_names, male_names, how='outer',on="name", indicator='common')

    female_names = female_names.loc[female_names['common'] != 'both']
    female_names = female_names.loc[female_names['common'] != 'right_only']
    female_names = female_names.drop(['gender_y', 'common'], axis=1)
    female_names.rename(columns={'gender_x': 'gender'}, inplace=True)

    merged_df = merged_df.drop(["key_0"], axis=1)

    merged_df = pd.merge(merged_df, female_names, left_on=merged_df['name_only'], right_on=female_names['name'], how='left')
    merged_df['gender'] = merged_df['gender_x'].fillna('') + merged_df['gender_y'].fillna('')
    merged_df = merged_df.drop(["name_only", "key_0", "Unnamed: 0", "name", "name_y", "gender_x", "gender_y"], axis=1)
    merged_df.rename(columns={'name_x': 'name'}, inplace=True)
    merged_df['gender'] = merged_df['gender'].replace('', np.nan)
    if save:
        pd.DataFrame.to_csv(merged_df, "cacd_meta/CACD_features_sex.csv")
    # print(merged_df.head())

    nan_count = merged_df['gender'].isna().sum()
    nan_rows = merged_df[merged_df['gender'].isna()]

    ungendered_names = set(nan_rows['name'].map(lambda x: x.split("_")[1]))
    # write_names_to_csv(ungendered_names, MANUAL_SAVE)

    annotated = pd.read_csv(MANUAL_READ)
    print(annotated.head())


if __name__ == "__main__":
    main()