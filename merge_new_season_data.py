"""
Merges new season dataset into big dataset
(To be run after new season predictions have been evaluated)
"""

import pandas as pd
import sys


def merge_new_into_old(
    new_season,
    data_folder="data/",
):
    # Load old data
    df_acting_old = pd.read_csv(data_folder + "oscardata_acting.csv")
    df_picture_old = pd.read_csv(data_folder + "oscardata_bestpicture.csv")
    df_director_old = pd.read_csv(data_folder + "oscardata_bestdirector.csv")

    # Load new data
    df_acting_new = pd.read_csv(
        data_folder + "oscardata_" + new_season + "_acting" + ".csv"
    )
    df_picture_new = pd.read_csv(
        data_folder + "oscardata_" + new_season + "_bestpicture" + ".csv"
    )
    df_director_new = pd.read_csv(
        data_folder + "oscardata_" + new_season + "_bestdirector" + ".csv"
    )

    if (
        df_acting_new["Winner"].isna().any()
        or df_picture_new["Winner"].isna().any()
        or df_director_new["Winner"].isna().any()
    ):
        sys.exit("Winner column not yet filled in")

    df_acting = pd.concat([df_acting_old, df_acting_new])
    df_picture = pd.concat([df_picture_old, df_picture_new])
    df_director = pd.concat([df_director_old, df_director_new])

    df_acting.to_csv(data_folder + "oscardata_acting.csv", index=False)
    print(df_acting.shape[0] - df_acting_old.shape[0], "new rows added")
    df_picture.to_csv(data_folder + "oscardata_bestpicture.csv", index=False)
    print(df_picture.shape[0] - df_picture_old.shape[0], "new rows added")
    df_director.to_csv(data_folder + "oscardata_bestdirector.csv", index=False)
    print(df_director.shape[0] - df_director_old.shape[0], "new rows added")


# Step 0: Fill in winners into oscardata_{new_season}_{category} .csvs

# Step 1: Run this script

merge_new_into_old("2024")
