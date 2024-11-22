import argparse

import pandas as pd

from src.data_scraping import (create_newseason_acting_dataframe,
                               create_newseason_director_dataframe,
                               create_newseason_picture_dataframe,
                               get_all_actor_data, get_all_award_data,
                               get_all_director_data, get_all_movie_data)
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser("scrape_data")
    parser.add_argument("--year", type=str, help="Year for which to scrape data")
    args = parser.parse_args()

    # Load new season nominations
    noms = pd.read_excel(f"data/nominations_{args.year}.xlsx")
    titles = noms["Film"].unique()
    actors = noms[noms["Category"].str.contains("actor|actress", case=False)][
        "Nominee"
    ].unique()
    directors = noms[noms["Category"] == "Director"]["Nominee"].unique()
    noms_director = noms[noms["Category"] == "Director"]
    noms_picture = noms[noms["Category"] == "Picture"]
    noms_actor = noms[noms["Category"].str.contains("actor|actress", case=False)]

    # Load config for scraping
    config = load_config(f"configs/scrape_config_{args.year}.yml")

    # Get (scrape) data on nominated movies, actors and directors
    movie_info_dict = get_all_movie_data(titles)
    actor_info_dict = get_all_actor_data(actors)
    director_info_dict = get_all_director_data(directors)

    # Get (scrape )info on award nominations and wins
    awards_info_dict = get_all_award_data(noms, config["urls"], config["maps"])

    # Create final dataframes for new season
    df_picture = create_newseason_picture_dataframe(
        noms_picture, movie_info_dict, awards_info_dict
    )
    df_director = create_newseason_director_dataframe(
        noms_director, movie_info_dict, director_info_dict, awards_info_dict
    )
    df_acting = create_newseason_acting_dataframe(
        noms_actor, movie_info_dict, actor_info_dict, awards_info_dict
    )

    # Save files
    df_picture.to_csv(f"data/oscardata_{args.year}_bestpicture-auto.csv", index=False)
    df_director.to_csv(f"data/oscardata_{args.year}_bestdirector-auto.csv", index=False)
    df_acting.to_csv(f"data/oscardata_{args.year}_acting-auto.csv", index=False)


if __name__ == "__main__":
    main()
