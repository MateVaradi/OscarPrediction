# Imports
import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import rottentomatoes as rt
from imdbmovies import IMDB


def quarter(date):
    month = int(date.split("-")[1])
    if month in [1, 2, 3]:
        q = 1
    elif month in [4, 5, 6]:
        q = 2
    elif month in [7, 8, 9]:
        q = 3
    elif month in [10, 11, 12]:
        q = 4
    else:
        q = np.nan
        print("WARNING: wrong date format")

    return q


def get_rt_score(score_func, movie_title, score_type):
    try:
        return int(score_func(movie_title))
    except:
        print(f"RT {score_type} score not found for: {movie_title}")
        return np.nan


def get_rt_ratings(movie_title):
    """
    Returns the Rotten Tomatoes critic score and audience score of a title
    """
    rt_critics_score = get_rt_score(rt.tomatometer, movie_title, "critic")
    rt_audience_score = get_rt_score(rt.audience_score, movie_title, "audience")
    return rt_critics_score, rt_audience_score


def get_person_birthyear(person_name):
    # Find person on IMDB
    try:
        imdb = IMDB()
        res = imdb.person_by_name(person_name)

        birth_year = res["mainEntity"]["birthDate"].split("-")[0]
        return birth_year
    except:
        print(f"WARNING: birth year information not found on IMDB for {person_name}")
        return np.nan


def get_person_oscar_history(person_name, oscar_df):
    person_oscar_history = oscar_df.loc[oscar_df.Nominee == person_name]
    oscar_noms = len(person_oscar_history)
    oscar_wins = person_oscar_history["Winner"].sum()
    return oscar_noms, oscar_wins


def get_actor_data(actor_name, oscar_actor_df):
    print(f"Getting actor data for {actor_name}")
    # Get birth year information
    birth_year = get_person_birthyear(actor_name)

    # Get award information
    oscar_noms, oscar_wins = get_person_oscar_history(actor_name, oscar_actor_df)

    return birth_year, oscar_noms, oscar_wins


def get_director_data(director_name, oscar_director_df):
    return get_person_oscar_history(director_name, oscar_director_df)


def get_imdb_movie_data(movie_title):
    """
    Returns IMDB rating, release quarter, genres, MPAA rating
    """
    try:
        imdb = IMDB()
        res = imdb.get_by_name(movie_title)

        rating = res["rating"].get("ratingValue", np.nan)
        release_date = res.get("datePublished", "")
        release_quarter = quarter(release_date) if release_date else np.nan
        genres = [g.lower() for g in res.get("genre", [])]
        mpaa = res.get("contentRating", np.nan)
        return rating, release_date, release_quarter, genres, mpaa

    except:
        print(f"WARNING: movie information not found on IMDB for {movie_title}")
        return np.nan, np.nan, np.nan, np.nan, np.nan


def get_movie_data(movie_title):
    movie_data = {}
    print(f"Getting movie data for {movie_title}")

    movie_data[movie_title] = {
        "IMDB_rating": None,
        "release_date": None,
        "release_quarter": None,
        "genres": None,
        "MPAA": None,
        "RT_critics": None,
        "RT_audience": None,
    }
    # Get data from IMDB
    imdb_data = get_imdb_movie_data(movie_title)
    movie_data[movie_title].update(
        dict(
            zip(
                ["IMDB_rating", "release_date", "release_quarter", "genres", "MPAA"],
                imdb_data,
            )
        )
    )

    # Get data from Rotten Tomatoes
    rt_data = get_rt_ratings(movie_title)
    movie_data[movie_title].update(dict(zip(["RT_critics", "RT_audience"], rt_data)))

    return movie_data


def get_all_movie_data(titles):
    movie_dict = {}
    for title in titles:
        movie_dict.update(get_movie_data(title))
    return movie_dict


def get_all_actor_data(names, acting_data_dir="data/oscardata_acting.csv"):
    df_acting = pd.read_csv(acting_data_dir)
    acting_data = df_acting[["Nominee", "Winner"]]

    return {
        name: dict(
            zip(
                ["birthyear", "oscar_nominations", "oscar_wins"],
                get_actor_data(name, acting_data),
            )
        )
        for name in names
    }


def get_all_director_data(names, director_data_dir="data/oscardata_bestdirector.csv"):
    df_director = pd.read_csv(director_data_dir)
    director_data = df_director[["Nominee", "Winner"]]

    return {
        name: dict(
            zip(
                ["oscar_nominations", "oscar_wins"],
                get_director_data(name, director_data),
            )
        )
        for name in names
    }


def parse_category(cat_nom_list):
    """
    Identify award category based on keywords in the list.
    """
    category_map = {
        "Director": "Best Director",
        "Picture": {"Best Film", "Best Picture"},
        "Actor": {"Best Actor in a Leading Role", "Best Actor"},
        "Actress": {"Best Actress in a Leading Role", "Best Actress"},
        "Supporting Actor": {
            "Best Actor in a Supporting Role",
            "Best Supporting Actor",
        },
        "Supporting Actress": {
            "Best Actress in a Supporting Role",
            "Best Supporting Actress",
        },
    }

    for cat, keywords in category_map.items():
        for keyword in keywords if isinstance(keywords, set) else [keywords]:
            if keyword in cat_nom_list:
                return cat, keyword
    return None, None


def clean_nominee(nominee, get_title=False):
    """
    Clean nominee string by removing references and extra characters.
    """
    nominee = re.sub(r"\[.*?\]", "", nominee)
    if get_title:
        return nominee.split(" – ")[1].strip()
    else:
        return nominee.split(" – ")[0].strip()


def extract_nominees(table, award, find_winner=True, category_location_map=None):
    """
    Extract nominees and winner for each category from a wiki table.
    """
    noms, wins = {}, {}

    if award.lower() in ["bafta", "critics choice", "cc"]:
        for i in range(3):
            for j in range(2):
                cat_nom_list = table[1].iloc[i, j].split("|")
                oscar_category, award_category = parse_category(cat_nom_list)
                if oscar_category:
                    noms[oscar_category] = [
                        clean_nominee(nom)
                        for nom in cat_nom_list
                        if nom and award_category not in nom
                    ]
                    if find_winner and noms[oscar_category]:
                        wins[oscar_category] = noms[oscar_category][0]

    elif award.lower() in ["pga", "dga"]:
        category = "Picture"
        cat_nom_list = table[2].iloc[0, 0].split("|")
        noms[category] = [
            clean_nominee(nom, get_title=True) for nom in cat_nom_list if nom
        ]
        wins[category] = noms[category][0]

    elif award.lower() in ["gg", "golden globe", "golden globes"]:
        for category, table_idx in category_location_map.items():
            row_ind, col_ind = map(int, table_idx.replace(" ", "").split(","))
            nom_list = table[1].iloc[row_ind, col_ind].split("|")
            noms[category] = [clean_nominee(nom) for nom in nom_list if nom]
            if find_winner and noms[category]:
                wins[category] = noms[category][0]

    elif award.lower() == "sag":
        for category, table_idx in category_location_map.items():
            row_ind, col_ind = map(int, table_idx.replace(" ", "").split(","))
            nom_str = table[2].iloc[row_ind, col_ind]
            nom_list = nom_str[nom_str.index("|") :].split(
                "|"
            )  # first is the category name
            noms[category] = [clean_nominee(nom) for nom in nom_list if nom]
            if find_winner and noms[category]:
                wins[category] = noms[category][0]

    return noms, wins


def get_award_data_from_wiki(
    wiki_url, award, find_winner=True, category_location_map=None
):
    """
    Retrieve nominees and winner information for specific categories from a Wikipedia page.
    """
    print(f"Getting {award} award data from: {wiki_url}")
    page = requests.get(wiki_url)
    table = pd.read_html(page.text.replace("\n", "|"))
    return extract_nominees(table, award, find_winner, category_location_map)


def get_num_total_oscar_noms(wiki_url, table_id):
    page = requests.get(wiki_url)
    table = pd.read_html(page.text.replace("\n", "|"))
    df_noms = table[table_id]
    for c in df_noms.columns:
        df_noms[c] = df_noms[c].str.replace("|", "")
        df_noms = df_noms.rename(columns={c: c.replace("|", "")})
    df_noms["Oscarstat_totalnoms"] = df_noms["Nominations"].astype(int)
    df_noms.drop("Nominations", axis="columns", inplace=True)
    return df_noms


def get_all_award_data(noms, award_wiki_urls, award_category_maps):
    awards = {}

    for award, wiki_url in award_wiki_urls.items():
        awards[award] = {}

        if award == "oscar":
            # Oscar - number of total nominations
            awards["oscar"]["num_noms"] = get_num_total_oscar_noms(
                wiki_url, award_category_maps["oscar"]["total_noms"]
            )

        else:
            awards[award]["noms"], awards[award]["wins"] = get_award_data_from_wiki(
                wiki_url,
                award=award,
                category_location_map=award_category_maps.get(award),
            )

    # add oscar nominated films and best director nominated films
    awards["oscar"]["noms"] = {}
    awards["oscar"]["noms"]["Director"] = (
        noms.loc[noms.Category == "Director", "Film"].unique().tolist()
    )
    awards["oscar"]["noms"]["Picture"] = (
        noms.loc[noms.Category == "Picture", "Film"].unique().tolist()
    )

    return awards


def create_newseason_picture_dataframe(noms_picture, movie_info_dict, awards_info_dict):

    movie_info_df = (
        pd.DataFrame.from_dict(movie_info_dict, orient="index")
        .reset_index()
        .rename(
            columns={
                "index": "Film",
                "RT_critics": "Rating_rtcritic",
                "RT_audience": "Rating_rtaudience",
                "IMDB_rating": "Rating_IMDB",
                "release_date": "Release_date",
            }
        )
    )
    df_picture = noms_picture.merge(movie_info_df, on="Film")

    # Add genre columns
    genre_cols = [
        "Genre_biography",
        "Genre_crime",
        "Genre_comedy",
        "Genre_drama",
        "Genre_horror",
        "Genre_fantasy",
        "Genre_sci-fi",
        "Genre_mystery",
        "Genre_music",
        "Genre_romance",
        "Genre_history",
        "Genre_war",
        "Genre_filmnoir",
        "Genre_thriller",
        "Genre_adventure",
        "Genre_family",
        "Genre_sport",
        "Genre_western",
        "Genre_action",
    ]
    genres = [g.replace("Genre_", "") for g in genre_cols]
    df_picture[genre_cols] = df_picture["genres"].apply(
        lambda x: pd.Series([(g in x) * 1 for g in genres])
    )
    df_picture = df_picture.drop("genres", axis=1)

    # Add MPAA columns
    mpaa_cats = ["G", "PG", "PG-13", "R", "NC-17"]
    df_picture["MPAA"] = pd.Categorical(df_picture["MPAA"], categories=mpaa_cats)
    mpaa_df = pd.get_dummies(df_picture["MPAA"], prefix="MPAA_", prefix_sep="") * 1
    df_picture = pd.concat([df_picture, mpaa_df], axis=1)
    df_picture = df_picture.rename(columns={"MPAA": "MPAA_rating"})

    # Add release date columns
    df_picture["release_quarter"] = pd.Categorical(
        df_picture["release_quarter"], categories=[1, 2, 3, 4]
    )
    release_df = (
        pd.get_dummies(df_picture["release_quarter"], prefix="Release_Q", prefix_sep="")
        * 1
    )
    df_picture = pd.concat([df_picture, release_df], axis=1)
    df_picture = df_picture.drop("release_quarter", axis=1)

    # Add award information

    # Oscar - Add number of total nominations
    df_oscar_noms = awards_info_dict["oscar"]["num_noms"]
    df_picture = df_picture.merge(df_oscar_noms, on="Film", how="left")
    df_picture["Oscarstat_totalnoms"] = df_picture["Oscarstat_totalnoms"].fillna(1)
    df_picture["Nom_Oscar_bestdirector"] = df_picture["Film"].apply(
        lambda x: x in awards_info_dict["oscar"]["noms"]["Director"]
    )

    # PGA - Best picture
    df_picture["Nom_PGA"] = df_picture["Film"].apply(
        lambda x: x in awards_info_dict["pga"]["noms"]["Picture"]
    )
    df_picture["Nonom_PGA"] = ~df_picture["Nom_PGA"]
    df_picture["Win_PGA"] = (
        df_picture["Film"] == awards_info_dict["pga"]["wins"]["Picture"]
    )
    df_picture["Nowin_PGA"] = ~df_picture["Win_PGA"]

    # DGA
    df_picture["Nom_DGA"] = df_picture["Film"].apply(
        lambda x: x in awards_info_dict["dga"]["noms"]["Picture"]
    )
    df_picture["Win_PGA"] = (
        df_picture["Film"] == awards_info_dict["dga"]["wins"]["Picture"]
    )

    # Critics Choice - Best Picture
    df_picture["Nom_Criticschoice"] = df_picture["Film"].apply(
        lambda x: x in awards_info_dict["cc"]["noms"]["Picture"]
    )
    df_picture["Nonom_Criticschoice"] = ~df_picture["Nom_Criticschoice"]
    df_picture["Win_Criticschoice"] = (
        df_picture["Film"] == awards_info_dict["cc"]["wins"]["Picture"]
    )
    df_picture["Nowin_Criticschoice"] = ~df_picture["Win_Criticschoice"]

    # SAG - Best cast
    df_picture["Nom_SAG_bestcast"] = df_picture["Film"].apply(
        lambda x: x in awards_info_dict["sag"]["noms"]["Picture"]
    )
    df_picture["Nonom_SAG_bestcast"] = ~df_picture["Nom_SAG_bestcast"]
    df_picture["Win_SAG_bestcast"] = (
        df_picture["Film"] == awards_info_dict["sag"]["wins"]["Picture"]
    )
    df_picture["Nowin_SAG_bestcast"] = ~df_picture["Win_SAG_bestcast"]

    # Golden Globes - Best Drama & Best Comedy
    df_picture["Nom_GoldenGlobe_bestdrama"] = df_picture["Film"].apply(
        lambda x: x in awards_info_dict["gg"]["noms"]["Best Drama"]
    )
    df_picture["Nom_GoldenGlobe_bestcomedy"] = df_picture["Film"].apply(
        lambda x: x in awards_info_dict["gg"]["noms"]["Best Comedy"]
    )
    df_picture["Win_GoldenGlobe_bestdrama"] = (
        df_picture["Film"] == awards_info_dict["gg"]["wins"]["Best Drama"]
    )
    df_picture["Win_GoldenGlobe_bestcomedy"] = (
        df_picture["Film"] == awards_info_dict["gg"]["wins"]["Best Comedy"]
    )

    # Bafta - Best picture
    df_picture["Nom_BAFTA"] = df_picture["Film"].apply(
        lambda x: x in awards_info_dict["bafta"]["noms"]["Picture"]
    )
    df_picture["Win_BAFTA"] = (
        df_picture["Film"] == awards_info_dict["bafta"]["wins"]["Picture"]
    )

    # Convert booleans to 0-1 columns
    nom_win_cols = [
        c
        for c in df_picture.columns
        if c.split("_")[0] in ["Nom", "Nonom", "Win", "Nowin"]
    ]
    for c in nom_win_cols:
        df_picture[c] = df_picture[c].astype(int)

    return df_picture


def create_newseason_director_dataframe(
    noms_director, movie_info_dict, director_info_dict, awards_info_dict
):

    # Add info on director
    director_info_df = (
        pd.DataFrame.from_dict(director_info_dict, orient="index")
        .reset_index()
        .rename(
            columns={
                "index": "Nominee",
                "birthyear": "Birthyear",
                "oscar_nominations": "Oscarstat_previousnominations_bestdirector",
                "oscar_wins": "Oscarstat_previouswins_bestdirector",
            }
        )
    )
    df_director = noms_director.merge(director_info_df, on="Nominee", how="left")

    # Add info on film
    movie_info_df = (
        pd.DataFrame.from_dict(movie_info_dict, orient="index")
        .reset_index()
        .rename(
            columns={
                "index": "Film",
                "RT_critics": "Rating_rtcritic",
                "RT_audience": "Rating_rtaudience",
                "IMDB_rating": "Rating_IMDB",
                "release_date": "Release_date",
            }
        )
    )

    # Add genre columns
    genre_cols = [
        "Genre_biography",
        "Genre_crime",
        "Genre_comedy",
        "Genre_drama",
        "Genre_horror",
        "Genre_fantasy",
        "Genre_sci-fi",
        "Genre_mystery",
        "Genre_music",
        "Genre_romance",
        "Genre_history",
        "Genre_war",
        "Genre_filmnoir",
        "Genre_thriller",
        "Genre_adventure",
        "Genre_family",
        "Genre_sport",
        "Genre_western",
        "Genre_action",
    ]
    genres = [g.replace("Genre_", "") for g in genre_cols]
    movie_info_df[genre_cols] = movie_info_df["genres"].apply(
        lambda x: pd.Series([(g in x) * 1 for g in genres])
    )
    movie_info_df = movie_info_df.drop("genres", axis=1)

    # Add MPAA columns
    mpaa_cats = ["G", "PG", "PG-13", "R", "NC-17"]
    movie_info_df["MPAA"] = pd.Categorical(movie_info_df["MPAA"], categories=mpaa_cats)
    mpaa_df = pd.get_dummies(movie_info_df["MPAA"], prefix="MPAA_", prefix_sep="") * 1
    movie_info_df = pd.concat([movie_info_df, mpaa_df], axis=1)
    movie_info_df = movie_info_df.rename(columns={"MPAA": "MPAA_rating"})

    # Add release date columns
    movie_info_df["release_quarter"] = pd.Categorical(
        movie_info_df["release_quarter"], categories=[1, 2, 3, 4]
    )
    release_df = (
        pd.get_dummies(
            movie_info_df["release_quarter"], prefix="Release_Q", prefix_sep=""
        )
        * 1
    )
    movie_info_df = pd.concat([movie_info_df, release_df], axis=1)
    movie_info_df = movie_info_df.drop("release_quarter", axis=1)

    df_director = df_director.merge(movie_info_df, on="Film", how="left")

    # Add award information

    # Oscar - Add number of total nominations
    df_oscar_noms = awards_info_dict["oscar"]["num_noms"]
    df_director = df_director.merge(df_oscar_noms, on="Film", how="left")
    df_director["Oscarstat_totalnoms"] = df_director["Oscarstat_totalnoms"].fillna(1)
    df_director["Nom_Oscar_bestfilm"] = df_director["Film"].apply(
        lambda x: x in awards_info_dict["oscar"]["noms"]["Picture"]
    )

    # Critics Choice - Best Picture
    df_director["Nom_Criticschoice"] = df_director["Nominee"].apply(
        lambda x: x in awards_info_dict["cc"]["noms"]["Director"]
    )
    df_director["Nonom_Criticschoice"] = ~df_director["Nom_Criticschoice"]
    df_director["Win_Criticschoice"] = (
        df_director["Nominee"] == awards_info_dict["cc"]["wins"]["Director"]
    )
    df_director["Nowin_Criticschoice"] = ~df_director["Win_Criticschoice"]

    # DGA
    df_director["Nom_DGA"] = df_director["Film"].apply(
        lambda x: x in awards_info_dict["dga"]["noms"]["Picture"]
    )
    df_director["Win_DGA"] = (
        df_director["Film"] == awards_info_dict["dga"]["wins"]["Picture"]
    )

    # Golden Globes - Best Director
    df_director["Nom_GoldenGlobe_bestdirector"] = df_director["Nominee"].apply(
        lambda x: x in awards_info_dict["gg"]["noms"]["Director"]
    )
    df_director["Win_GoldenGlobe_bestdirector"] = (
        df_director["Nominee"] == awards_info_dict["gg"]["wins"]["Director"]
    )

    # Bafta - Best Director
    df_director["Nom_BAFTA"] = df_director["Nominee"].apply(
        lambda x: x in awards_info_dict["bafta"]["noms"]["Director"]
    )
    df_director["Win_BAFTA"] = (
        df_director["Nominee"] == awards_info_dict["bafta"]["wins"]["Director"]
    )

    # Convert booleans to 0-1 columns
    nom_win_cols = [
        c
        for c in df_director.columns
        if c.split("_")[0] in ["Nom", "Nonom", "Win", "Nowin"]
    ]
    for c in nom_win_cols:
        df_director[c] = df_director[c].astype(int)

    return df_director


def is_actor_nominee(nom_df, awards_info_dict, award):
    return nom_df[["Category", "Nominee"]].apply(
        lambda x: x["Nominee"] in awards_info_dict[award]["noms"][x["Category"]], axis=1
    )


def is_actor_golden_globe_nominee(nom_df, awards_info_dict, gg_category):
    gg_category = gg_category.capitalize()
    if gg_category in ["Drama", "Comedy"]:
        return nom_df[["Category", "Nominee"]].apply(
            lambda x: (
                np.nan
                if "Supporting" in x["Category"]
                else x["Nominee"]
                in awards_info_dict["gg"]["noms"][f'{gg_category} {x["Category"]}']
            ),
            axis=1,
        )
    elif gg_category == "Supporting":
        return nom_df[["Category", "Nominee"]].apply(
            lambda x: (
                np.nan
                if "Supporting" not in x["Category"]
                else x["Nominee"] in awards_info_dict["gg"]["noms"][x["Category"]]
            ),
            axis=1,
        )
    else:
        raise ValueError(f"gg_category: {gg_category} is not recognized")


def is_actor_winner(nom_df, awards_info_dict, award):
    return nom_df[["Category", "Nominee"]].apply(
        lambda x: x["Nominee"] == awards_info_dict[award]["wins"][x["Category"]], axis=1
    )


def is_actor_golden_globe_winner(nom_df, awards_info_dict, gg_category):
    gg_category = gg_category.capitalize()
    if gg_category in ["Drama", "Comedy"]:
        return nom_df[["Category", "Nominee"]].apply(
            lambda x: (
                np.nan
                if "Supporting" in x["Category"]
                else x["Nominee"]
                == awards_info_dict["gg"]["wins"][f'{gg_category} {x["Category"]}']
            ),
            axis=1,
        )
    elif gg_category == "Supporting":
        return nom_df[["Category", "Nominee"]].apply(
            lambda x: (
                np.nan
                if "Supporting" not in x["Category"]
                else x["Nominee"] == awards_info_dict["gg"]["wins"][x["Category"]]
            ),
            axis=1,
        )
    else:
        raise ValueError(f"gg_category: {gg_category} is not recognized")


def create_newseason_acting_dataframe(
    noms_acting, movie_info_dict, actor_info_dict, awards_info_dict
):

    # Add info on actors
    actor_info_df = (
        pd.DataFrame.from_dict(actor_info_dict, orient="index")
        .reset_index()
        .rename(
            columns={
                "index": "Nominee",
                "birthyear": "Birthyear",
                "oscar_nominations": "Oscarstat_previousnominations_acting",
                "oscar_wins": "Oscarstat_previouswins_acting",
            }
        )
    )
    actor_info_df["Oscarstat_previousnominee_acting"] = (
        actor_info_df["Oscarstat_previousnominations_acting"] > 0
    ) * 1
    actor_info_df["Oscarstat_previouswinner_acting"] = (
        actor_info_df["Oscarstat_previouswins_acting"] > 0
    ) * 1

    # Calculate age
    this_year = noms_acting["Year"].max()
    actor_info_df["Age"] = this_year - actor_info_df["Birthyear"].astype(int)
    age_cols = [
        "Age_[0-25]",
        "Age_[25-35]",
        "Age_[35-45]",
        "Age_[45-55]",
        "Age_[55-65]",
        "Age_[65-75]",
        "Age_[75+]",
    ]
    for age_col in age_cols:
        age_bucket_bounds = list(map(int, re.findall(r"\d+", age_col))) + [np.inf]
        lb = age_bucket_bounds[0]
        ub = age_bucket_bounds[1]
        actor_info_df[age_col] = actor_info_df["Age"].apply(
            lambda x: (lb < x and x <= ub) * 1
        )

    df_acting = noms_acting.merge(actor_info_df, on="Nominee", how="left")
    df_acting["Female"] = df_acting["Category"].apply(
        lambda x: int("actress" in str(x).lower())
    )

    # Add info on film
    movie_info_df = (
        pd.DataFrame.from_dict(movie_info_dict, orient="index")
        .reset_index()
        .rename(
            columns={
                "index": "Film",
                "RT_critics": "Rating_rtcritic",
                "RT_audience": "Rating_rtaudience",
                "IMDB_rating": "Rating_IMDB",
                "release_date": "Release_date",
            }
        )
    )

    # Add genre columns
    genre_cols = [
        "Genre_action",
        "Genre_adventure",
        "Genre_biography",
        "Genre_crime",
        "Genre_comedy",
        "Genre_drama",
        "Genre_horror",
        "Genre_fantasy",
        "Genre_sci-fi",
        "Genre_mystery",
        "Genre_music",
        "Genre_romance",
        "Genre_history",
        "Genre_war",
        "Genre_filmnoir",
        "Genre_thriller",
        "Genre_family",
        "Genre_sport",
        "Genre_western",
    ]
    genres = [g.replace("Genre_", "") for g in genre_cols]
    movie_info_df[genre_cols] = movie_info_df["genres"].apply(
        lambda x: pd.Series([(g in x) * 1 for g in genres])
    )
    movie_info_df = movie_info_df.drop("genres", axis=1)

    # Add MPAA columns
    mpaa_cats = ["G", "PG", "PG-13", "R", "NC-17"]
    movie_info_df["MPAA"] = pd.Categorical(movie_info_df["MPAA"], categories=mpaa_cats)
    mpaa_df = pd.get_dummies(movie_info_df["MPAA"], prefix="MPAA_", prefix_sep="") * 1
    movie_info_df = pd.concat([movie_info_df, mpaa_df], axis=1)
    movie_info_df = movie_info_df.rename(columns={"MPAA": "MPAA_rating"})

    # Add release date columns
    movie_info_df["release_quarter"] = pd.Categorical(
        movie_info_df["release_quarter"], categories=[1, 2, 3, 4]
    )
    release_df = (
        pd.get_dummies(
            movie_info_df["release_quarter"], prefix="Release_Q", prefix_sep=""
        )
        * 1
    )
    movie_info_df = pd.concat([movie_info_df, release_df], axis=1)
    movie_info_df = movie_info_df.drop("release_quarter", axis=1)

    df_acting = df_acting.merge(movie_info_df, on="Film", how="left")

    # Add award information

    # Oscar - Add number of total nominations
    df_oscar_noms = awards_info_dict["oscar"]["num_noms"]
    df_acting = df_acting.merge(df_oscar_noms, on="Film", how="left")
    df_acting["Oscarstat_totalnoms"] = df_acting["Oscarstat_totalnoms"].fillna(1)
    df_acting["Nom_Oscar_bestfilm"] = df_acting["Film"].apply(
        lambda x: x in awards_info_dict["oscar"]["noms"]["Picture"]
    )

    # Critics Choice
    df_acting["Nom_Criticschoice"] = is_actor_nominee(df_acting, awards_info_dict, "cc")
    df_acting["Nonom_Criticschoice"] = ~df_acting["Nom_Criticschoice"]
    df_acting["Win_Criticschoice"] = is_actor_winner(df_acting, awards_info_dict, "cc")
    df_acting["Nowin_Criticschoice"] = ~df_acting["Win_Criticschoice"]

    # SAG
    df_acting["Nom_SAG_acting"] = is_actor_nominee(df_acting, awards_info_dict, "sag")
    df_acting["Nonom_SAG_acting"] = ~df_acting["Nom_SAG_acting"]
    df_acting["Win_SAG_acting"] = is_actor_winner(df_acting, awards_info_dict, "sag")
    df_acting["Nowin_SAG_acting"] = ~df_acting["Win_SAG_acting"]
    df_acting["Nom_SAG_bestcast"] = df_acting["Film"].apply(
        lambda x: x in awards_info_dict["sag"]["noms"]["Picture"]
    )
    df_acting["Nonom_SAG_bestcast"] = ~df_acting["Nom_SAG_bestcast"]
    df_acting["Win_SAG_bestcast"] = df_acting["Film"].apply(
        lambda x: x == awards_info_dict["sag"]["wins"]["Picture"]
    )
    df_acting["Nowin_SAG_bestcast"] = ~df_acting["Win_SAG_bestcast"]

    # Golden Globes
    df_acting["Nom_GoldenGlobe_comedy-leadacting"] = is_actor_golden_globe_nominee(
        df_acting, awards_info_dict, "comedy"
    )
    df_acting["Win_GoldenGlobe_comedy-leadacting"] = is_actor_golden_globe_winner(
        df_acting, awards_info_dict, "comedy"
    )
    df_acting["Nom_GoldenGlobe_drama-leadacting"] = is_actor_golden_globe_nominee(
        df_acting, awards_info_dict, "drama"
    )
    df_acting["Win_GoldenGlobe_drama-leadacting"] = is_actor_golden_globe_winner(
        df_acting, awards_info_dict, "drama"
    )
    df_acting["Nom_GoldenGlobe_supportingacting"] = is_actor_golden_globe_nominee(
        df_acting, awards_info_dict, "supporting"
    )
    df_acting["Win_GoldenGlobe_supportingacting"] = is_actor_golden_globe_winner(
        df_acting, awards_info_dict, "supporting"
    )

    # Bafta
    df_acting["Nom_BAFTA"] = is_actor_nominee(df_acting, awards_info_dict, "bafta")
    df_acting["Win_BAFTA"] = is_actor_winner(df_acting, awards_info_dict, "bafta")

    # Convert booleans to 0-1 columns
    nom_win_cols = [
        c
        for c in df_acting.columns
        if c.split("_")[0] in ["Nom", "Nonom", "Win", "Nowin"]
    ]
    for c in nom_win_cols:
        df_acting[c] = df_acting[c].apply(lambda x: int(x) if not pd.isnull(x) else x)

    return df_acting
