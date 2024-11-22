"""
Merges new season dataset into big dataset
(To be run after new season predictions have been evaluated)
"""

import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser("merge_new_season_data")
    parser.add_argument("--year", type=str, help="Year for which to scrape data")
    args = parser.parse_args()

    # Load old data
    df_acting_old = pd.read_csv("data/oscardata_acting.csv")
    df_picture_old = pd.read_csv("data/oscardata_bestpicture.csv")
    df_director_old = pd.read_csv("data/oscardata_bestdirector.csv")

    # Load new data
    df_acting_new = pd.read_csv("data/oscardata_" + args.year + "_acting" + ".csv")
    df_picture_new = pd.read_csv(
        "data/oscardata_" + args.year + "_bestpicture" + ".csv"
    )
    df_director_new = pd.read_csv(
        "data/oscardata_" + args.year + "_bestdirector" + ".csv"
    )

    if (
        df_acting_new["Winner"].isna().any()
        or df_picture_new["Winner"].isna().any()
        or df_director_new["Winner"].isna().any()
    ):
        raise ValueError("Winner column not yet filled in")

    # Check if all required columns exist
    missing_cols_acting = set(df_acting_old.columns).difference(
        set(df_acting_new.columns)
    )
    missing_cols_director = set(df_director_old.columns).difference(
        set(df_director_new.columns)
    )
    missing_cols_picture = set(df_picture_old.columns).difference(
        set(df_picture_new.columns)
    )

    if missing_cols_acting:
        raise ValueError(
            f"Missing columns for acting data: {', '.join(missing_cols_acting)}"
        )
    if missing_cols_director:
        raise ValueError(
            f"Missing columns for director data: {', '.join(missing_cols_director)}"
        )
    if missing_cols_picture:
        raise ValueError(
            f"Missing columns for picture data: {', '.join(missing_cols_picture)}"
        )

    # Check if new variables are added
    new_cols_acting = set(df_acting_new.columns).difference(set(df_acting_old.columns))
    new_cols_director = set(df_director_new.columns).difference(
        set(df_director_old.columns)
    )
    new_cols_picture = set(df_picture_new.columns).difference(
        set(df_picture_old.columns)
    )

    if new_cols_acting:
        print(f"New columns added to acting data: {', '.join(new_cols_acting)}")
    if new_cols_director:
        print(f"New columns added to director data: {', '.join(new_cols_director)}")
    if new_cols_picture:
        print(f"New columns added to picture data: {', '.join(new_cols_picture)}")

    # Check whether data had been added before:
    if str(args.year) in df_acting_old["Year"].unique():
        raise ValueError("Year had already been added to acting data.")
    if str(args.year) in df_director_old["Year"].unique():
        raise ValueError("Year had already been added to director data.")
    if str(args.year) in df_picture_old["Year"].unique():
        raise ValueError("Year had already been added to picture data.")

    df_acting = pd.concat([df_acting_old, df_acting_new])
    df_picture = pd.concat([df_picture_old, df_picture_new])
    df_director = pd.concat([df_director_old, df_director_new])

    df_acting.to_csv("data/oscardata_acting.csv", index=False)
    print(df_acting.shape[0] - df_acting_old.shape[0], "new rows added")
    df_picture.to_csv("data/oscardata_bestpicture.csv", index=False)
    print(df_picture.shape[0] - df_picture_old.shape[0], "new rows added")
    df_director.to_csv("data/oscardata_bestdirector.csv", index=False)
    print(df_director.shape[0] - df_director_old.shape[0], "new rows added")


if __name__ == "__main__":

    # Step 0: Fill in winners into oscardata_{new_season}_{category}.csvs

    # Step 1: Run this script
    main()
