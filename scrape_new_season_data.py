"""
Scrapes data from the internet for nominees of the new season
"""

# Imports
from rotten_tomatoes_scraper.rt_scraper import MovieScraper
from imdb import Cinemagoer
import pandas as pd
import numpy as np
import re
from time import sleep
from random import randint
from datetime import datetime

# Settings
pd.set_option("display.max_columns", 100)

""" Helper functions """


def quarter(date):
    month = date.split(" ")[1]
    if month in ["Jan", "Feb", "Mar"]:
        q = 1
    elif month in ["Apr", "May", "Jun"]:
        q = 2
    elif month in ["Jul", "Aug", "Sep"]:
        q = 3
    elif month in ["Oct", "Nov", "Dec"]:
        q = 4
    else:
        print("wrong date format")
    return q


""" Main functions """


# Old function
def get_RT_ratings(movie_title):
    """
    Returns the Rotten Tomatoes critic score and audience score of a title
    """

    # Extract URL
    RT_search = MovieScraper()
    search_res = RT_search.search(movie_title)

    # Exact match
    url_list = [
        movie_dict["url"]
        for movie_dict in search_res["movies"]
        if movie_dict["name"].lower() == movie_title.lower()
    ]
    if len(url_list) == 1:
        url = url_list[0]
    # No exact match -  return the latest one
    elif not url_list:
        url_list = sorted(
            [
                (movie_dict["url"], movie_dict["year"])
                for movie_dict in search_res["movies"]
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        url = url_list[0][0]
        print(f"No exact match found. Going with {url}")
    # More than one exact match - return the latest one
    elif len(url_list) > 1:
        url_list = sorted(
            [
                (movie_dict["url"], movie_dict["year"])
                for movie_dict in search_res["movies"]
                if movie_dict["name"].lower() == movie_title.lower()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        url = url_list[0][0]
        print(f"More than one exact match found. Going with {url}")

    movie_scraper = MovieScraper(movie_url="https://www.rottentomatoes.com" + url)
    movie_scraper.extract_metadata()
    rt_critics_score = int(movie_scraper.metadata["Score_Rotten"])
    rt_audience_score = int(movie_scraper.metadata["Score_Audience"])
    return rt_critics_score, rt_audience_score


def get_RT_ratings(movie_title):
    """
    Returns the Rotten Tomatoes critic score and audience score of a title
    """

    if movie_title == "Im Westen nichts Neues":
        url = "all_quiet_on_the_western_front_2022"
    elif movie_title == "Tár":
        url = "tar_2022"
    elif movie_title == "Living":
        url = "living_2022"
    else:
        url = movie_title.replace(":", "").replace(" ", "_").lower()

    movie_scraper = MovieScraper(movie_url="https://www.rottentomatoes.com/m/" + url)
    movie_scraper.extract_metadata()
    rt_critics_score = int(movie_scraper.metadata["Score_Rotten"])
    rt_audience_score = int(movie_scraper.metadata["Score_Audience"])
    return rt_critics_score, rt_audience_score


def get_IMDB_movie_data(movie_title):
    """
    Returns the following information of a title:
        IMDB rating
        release quarter
        genres
        MPAA rating
        awards
    """
    # Get movie data from IMDB API
    ia = Cinemagoer()
    res = ia._search_movie(movie_title, results=True)
    movie_ID = res[0][0]
    if res[0][1]["title"] != movie_title:
        print("Titles do not exactly match: ", movie_title, res[0][1]["title"])
    movie = ia.get_movie(movie_ID)
    rating = movie.data["rating"]
    release_date = movie.data["original air date"].split(" (")[0]
    try:
        release_quarter = quarter(release_date)
    except:  # when release month not (yet) specified
        release_quarter = 1
    genres = [genre.lower() for genre in movie.data["genres"]]
    MPAA = [
        c.split(":")[1]
        for c in movie.data["certificates"]
        if (c.startswith("United States") or "USA" in c)
    ][0]
    awards = dict()
    award_data = ia.get_movie_awards(movie_ID)["data"]["awards"]
    for nom in award_data:

        if nom["award"] == "PGA Award":
            if "PGA" not in awards.keys():
                awards["PGA"] = {"categories": [], "results": [], "to": []}
            if "notes" in nom.keys():
                awards["PGA"]["categories"].append(nom["notes"])
            else:
                awards["PGA"]["categories"].append(nom["category"])
            awards["PGA"]["results"].append(nom["result"])
            if "to" in nom.keys():
                awards["PGA"]["to"].append([str(n) for n in nom["to"]])
            else:
                awards["PGA"]["to"].append([])

        elif nom["award"] == "BAFTA Film Award":
            if "BAFTA" not in awards.keys():
                awards["BAFTA"] = {"categories": [], "results": [], "to": []}
            if "notes" in nom.keys():
                awards["BAFTA"]["categories"].append(nom["notes"])
            else:
                awards["BAFTA"]["categories"].append(nom["category"])
            awards["BAFTA"]["results"].append(nom["result"])
            if "to" in nom.keys():
                awards["BAFTA"]["to"].append([str(n) for n in nom["to"]])
            else:
                awards["BAFTA"]["to"].append([])

        elif nom["award"] == "Golden Globe":
            if "Golden Globe" not in awards.keys():
                awards["Golden Globe"] = {"categories": [], "results": [], "to": []}
            if "notes" in nom.keys():
                awards["Golden Globe"]["categories"].append(nom["notes"])
            else:
                awards["Golden Globe"]["categories"].append(nom["category"])
            awards["Golden Globe"]["results"].append(nom["result"])
            if "to" in nom.keys():
                awards["Golden Globe"]["to"].append([str(n) for n in nom["to"]])
            else:
                awards["Golden Globe"]["to"].append([])

        elif nom["award"] in ["Academy Awards, USA", "Oscar"]:
            if "Oscar" not in awards.keys():
                awards["Oscar"] = {"categories": [], "results": [], "to": []}
            if "notes" in nom.keys():
                awards["Oscar"]["categories"].append(nom["notes"])
            else:
                awards["Oscar"]["categories"].append(nom["category"])
            awards["Oscar"]["results"].append(nom["result"])
            if "to" in nom.keys():
                awards["Oscar"]["to"].append([str(n) for n in nom["to"]])
            else:
                awards["Oscar"]["to"].append([])

        elif nom["award"] == "Critics Choice Award":
            if "Critics Choice" not in awards.keys():
                awards["Critics Choice"] = {"categories": [], "results": [], "to": []}
            if "notes" in nom.keys():
                awards["Critics Choice"]["categories"].append(nom["notes"])
            else:
                awards["Critics Choice"]["categories"].append(nom["category"])
            awards["Critics Choice"]["results"].append(nom["result"])
            if "to" in nom.keys():
                awards["Critics Choice"]["to"].append([str(n) for n in nom["to"]])
            else:
                awards["Critics Choice"]["to"].append([])

        elif nom["award"] == "DGA Award":
            if "DGA" not in awards.keys():
                awards["DGA"] = {"categories": [], "results": [], "to": []}
            if "notes" in nom.keys():
                awards["DGA"]["categories"].append(nom["notes"])
            else:
                awards["DGA"]["categories"].append(nom["category"])
            awards["DGA"]["results"].append(nom["result"])
            if "to" in nom.keys():
                awards["DGA"]["to"].append([str(n) for n in nom["to"]])
            else:
                awards["DGA"]["to"].append([])

        elif (nom["category"] == "Screen Actors Guild Awards") or (
            nom["award"] == "Screen Actors Guild Awards"
        ):
            if "SAG" not in awards.keys():
                awards["SAG"] = {"categories": [], "results": [], "to": []}
            if "notes" in nom.keys():
                awards["SAG"]["categories"].append(nom["notes"])
            else:
                awards["SAG"]["categories"].append(nom["category"])
            awards["SAG"]["results"].append(nom["result"])
            if "to" in nom.keys():
                awards["SAG"]["to"].append([str(n) for n in nom["to"]])
            else:
                awards["SAG"]["to"].append([])

    return rating, release_date, release_quarter, genres, MPAA, awards


def get_actor_data(actor_name):
    # Get actor data from IMDB API
    ia = Cinemagoer()
    res = ia._search_person(actor_name, results=True)
    person_ID = res[0][0]
    found_name = " ".join(res[0][1]["name"].split(",")[::-1]).strip(" ")
    if found_name.lower() != actor_name.lower():
        print("Titles do not exactly match: ", actor_name, found_name)
    person = ia.get_person(person_ID)
    if "birth date" in person.data.keys():
        birthyear = int(person.data["birth date"].split("-")[0])
        age = datetime.today().year - birthyear
    else:
        birthyear = np.nan
        age = np.nan
    award_data = ia.get_person_awards(person_ID)["data"]["awards"]
    oscar_nominations = oscar_wins = 0
    for nom in award_data:
        if (nom["award"] == "Academy Awards, USA") and (
            nom["category"]
            in [
                "Best Performance by an Actress in a Leading Role",
                "Best Performance by an Actress in a Supporting Role",
                "Best Performance by an Actor in a Leading Role",
                "Best Performance by an Actor in a Supporting Role",
                "Best Actor in a Leading Role",
                "Best Actor in a Supporting Role",
                "Best Actress in a Leading Role",
                "Best Actress in a Supporting Role",
            ]
        ):
            oscar_nominations += 1
            if "result" in nom.keys():
                if nom["result"] == "Winner":
                    oscar_wins += 1

    return age, birthyear, oscar_nominations, oscar_wins


def get_director_data(director_name):
    # Get director data from IMDB API
    ia = Cinemagoer()
    res = ia._search_person(director_name, results=True)
    person_ID = res[0][0]
    found_name = " ".join(res[0][1]["name"].split(",")[::-1]).strip(" ")
    if found_name.lower() != director_name.lower():
        print("Titles do not exactly match: ", director_name, found_name)
    award_data = ia.get_person_awards(person_ID)["data"]["awards"]
    oscar_nominations = oscar_wins = 0
    for nom in award_data:
        if (nom["award"] == "Academy Awards, USA") and (
            nom["category"] in ["Best Director", "Best Achievement in Directing"]
        ):
            oscar_nominations += 1
            if "result" in nom.keys():
                if nom["result"] == "Winner":
                    oscar_wins += 1

    return oscar_nominations, oscar_wins


def get_all_movie_data(titles):
    new_movie_data = {}
    for i, movie_title in enumerate(titles):
        movie_title = str(movie_title)
        print(f"Getting data for {movie_title}")
        new_movie_data[movie_title] = {}

        # Get data from IMDB
        tmp = get_IMDB_movie_data(movie_title)
        new_movie_data[movie_title]["IMDB_rating"] = tmp[0]
        new_movie_data[movie_title]["release_date"] = tmp[1]
        new_movie_data[movie_title]["release_quarter"] = tmp[2]
        new_movie_data[movie_title]["genres"] = tmp[3]
        new_movie_data[movie_title]["MPAA"] = tmp[4]
        new_movie_data[movie_title]["awards"] = tmp[5]
        # Get data from RT

        # sometimes the scrapers may fail and we need to do some hardcoded changes to make it work, for example
        if movie_title == "Parasite":
            movie_title_RT = "Parasite (Gisaengchung)"
        else:
            movie_title_RT = movie_title

        if movie_title == "Tick, Tick... Boom!":
            new_movie_data[movie_title]["RT_critics"] = 88
            new_movie_data[movie_title]["RT_audience"] = 96
        elif movie_title == "Don't Look Up":
            new_movie_data[movie_title]["RT_critics"] = 55
            new_movie_data[movie_title]["RT_audience"] = 78
        else:
            tmp = get_RT_ratings(movie_title_RT)
            new_movie_data[movie_title]["RT_critics"] = tmp[0]
            new_movie_data[movie_title]["RT_audience"] = tmp[1]

        if i > 7:
            i = 0
            sleep(randint(8, 20))

    return new_movie_data


def get_all_actor_data(names):
    new_actor_data = dict()
    for name in names:
        print(f"Getting data for {name}")
        new_actor_data[name] = {}
        tmp = get_actor_data(name)
        new_actor_data[name]["age"] = tmp[0]
        new_actor_data[name]["birthyear"] = tmp[1]
        new_actor_data[name]["oscar_nominations"] = tmp[2]
        new_actor_data[name]["oscar_wins"] = tmp[3]

    return new_actor_data


def get_all_director_data(names):
    new_director_data = dict()
    for name in names:
        print(f"Getting data for {name}")
        new_director_data[name] = {}
        tmp = get_director_data(name)
        new_director_data[name]["oscar_nominations"] = tmp[0]
        new_director_data[name]["oscar_wins"] = tmp[1]

    return new_director_data


def get_all_newseason_data(new_season):
    df = pd.read_excel(f"data/nominations_{new_season}.xlsx")

    titles = df["Film"].unique()
    actors = df[
        df["Category"].apply(
            lambda x: ("actor" in x.lower()) or ("actress") in x.lower()
        )
    ]["Nominee"].unique()
    directors = df[df["Category"] == "Director"]["Nominee"].unique()

    # Scrape movie data in chunks of 5
    nominated_movies = dict()
    for title_chunk in [titles[i : i + 5] for i in range(0, len(titles), 5)]:
        nominated_movies.update(get_all_movie_data(title_chunk))
    nominated_actors = get_all_actor_data(actors)
    nominated_directors = get_all_director_data(directors)

    return nominated_movies, nominated_actors, nominated_directors


def create_newseason_picture_dataframe(nominated_movies, new_season, suffix=""):
    print(f"Creating {new_season} dataframe for Best Picture")

    df = pd.read_excel(f"data/nominations_{new_season}.xlsx")
    df = df[df["Category"] == "Picture"]
    df["Film"] = df["Film"].astype(str)

    start_cols = df.columns
    final_cols = pd.read_csv(f"data/oscardata_bestpicture.csv").columns
    for col in final_cols:
        if col not in start_cols:
            df[col] = np.nan

    # Rating columns
    df["Rating_IMDB"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["IMDB_rating"]
            for movie in nominated_movies.keys()
        }
    )
    df["Rating_rtcritic"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["RT_critics"]
            for movie in nominated_movies.keys()
        }
    )
    df["Rating_rtaudience"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["RT_audience"]
            for movie in nominated_movies.keys()
        }
    )

    # Oscar stat columns
    df["Oscarstat_totalnoms"] = df["Film"].map(
        {
            movie: len(nominated_movies[movie]["awards"]["Oscar"]["categories"])
            for movie in nominated_movies.keys()
        }
    )

    # Genre columns
    genre_cols = [col for col in final_cols if "Genre" in col]
    for genre in genre_cols:
        df[genre] = df["Film"].map(
            {
                movie: (genre.split("_")[1] in nominated_movies[movie]["genres"]) * 1
                for movie in nominated_movies.keys()
            }
        )

    # MPAA columns
    MPAA_rating_types = [
        col for col in final_cols if ("MPAA" in col) and ("rating" not in col)
    ]
    df["MPAA_rating"] = df["Film"].map(
        {movie: nominated_movies[movie]["MPAA"] for movie in nominated_movies.keys()}
    )
    df[MPAA_rating_types] = 0
    for rating_type in MPAA_rating_types:
        df.loc[df["MPAA_rating"] == rating_type.split("_")[1], rating_type] = 1

    # Release columns
    df["Release_date"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["release_date"]
            for movie in nominated_movies.keys()
        }
    )
    releaseQ_cols = [col for col in final_cols if "Release_Q" in col]
    for Q in releaseQ_cols:
        Q_num = int(Q[-1])
        df[Q] = df["Film"].map(
            {
                movie: (nominated_movies[movie]["release_quarter"] == Q_num) * 1
                for movie in nominated_movies.keys()
            }
        )

    # Nom and Win columns
    # Oscar
    df["Nom_Oscar_bestdirector"] = df["Film"].map(
        {
            movie: any(
                dir_cat in nominated_movies[movie]["awards"]["Oscar"]["categories"]
                for dir_cat in ["Best Director", "Best Achievement in Directing"]
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    # DGA
    df["Nom_DGA"] = df["Film"].map(
        {
            movie: ("DGA" in nominated_movies[movie]["awards"].keys()) * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Win_DGA"] = df["Film"].map(
        {
            movie: (
                ("DGA" in nominated_movies[movie]["awards"].keys())
                and (nominated_movies[movie]["awards"]["DGA"]["results"][0] == "Winner")
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    # BAFTA
    df["Nom_BAFTA"] = df["Film"].map(
        {
            movie: (
                ("BAFTA" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Film"
                    in nominated_movies[movie]["awards"]["BAFTA"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Win_BAFTA"] = df["Film"].map(
        {
            movie: (
                ("BAFTA" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Film"
                    in nominated_movies[movie]["awards"]["BAFTA"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["BAFTA"]["categories"].index(
                        "Best Film"
                    )
                )
                and (
                    nominated_movies[movie]["awards"]["BAFTA"]["results"][
                        nominated_movies[movie]["awards"]["BAFTA"]["categories"].index(
                            "Best Film"
                        )
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    # Golden Globe
    df["Nom_GoldenGlobe_bestdrama"] = df["Film"].map(
        {
            movie: (
                ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Motion Picture - Drama"
                    in nominated_movies[movie]["awards"]["Golden Globe"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Nom_GoldenGlobe_bestcomedy"] = df["Film"].map(
        {
            movie: (
                ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Motion Picture - Musical or Comedy"
                    in nominated_movies[movie]["awards"]["Golden Globe"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Win_GoldenGlobe_bestdrama"] = df["Film"].map(
        {
            movie: (
                ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Motion Picture - Drama"
                    in nominated_movies[movie]["awards"]["Golden Globe"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["Golden Globe"]["results"][
                        nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ].index("Best Motion Picture - Drama")
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Win_GoldenGlobe_bestcomedy"] = df["Film"].map(
        {
            movie: (
                ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Motion Picture - Musical or Comedy"
                    in nominated_movies[movie]["awards"]["Golden Globe"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["Golden Globe"]["results"][
                        nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ].index("Best Motion Picture - Musical or Comedy")
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    # SAG
    df["Nom_SAG_bestcast"] = df["Film"].map(
        {
            movie: (
                ("SAG" in nominated_movies[movie]["awards"].keys())
                and (
                    "Outstanding Performance by a Cast in a Motion Picture"
                    in nominated_movies[movie]["awards"]["SAG"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Nonom_SAG_bestcast"] = (
        df["Nom_SAG_bestcast"].astype(bool).apply(lambda x: int(not x))
    )

    df["Win_SAG_bestcast"] = df["Film"].map(
        {
            movie: (
                ("SAG" in nominated_movies[movie]["awards"].keys())
                and (
                    "Outstanding Performance by a Cast in a Motion Picture"
                    in nominated_movies[movie]["awards"]["SAG"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["SAG"]["results"][
                        nominated_movies[movie]["awards"]["SAG"]["categories"].index(
                            "Outstanding Performance by a Cast in a Motion Picture"
                        )
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Nowin_SAG_bestcast"] = (
        df["Win_SAG_bestcast"].astype(bool).apply(lambda x: int(not x))
    )

    # PGA
    df["Nom_PGA"] = df["Film"].map(
        {
            movie: (
                ("PGA" in nominated_movies[movie]["awards"].keys())
                and (
                    "Outstanding Producer of Theatrical Motion Pictures"
                    in nominated_movies[movie]["awards"]["PGA"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Nonom_PGA"] = df["Nom_PGA"].astype(bool).apply(lambda x: int(not x))

    df["Win_PGA"] = df["Film"].map(
        {
            movie: (
                ("PGA" in nominated_movies[movie]["awards"].keys())
                and (
                    "Outstanding Producer of Theatrical Motion Pictures"
                    in nominated_movies[movie]["awards"]["PGA"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["PGA"]["results"][
                        nominated_movies[movie]["awards"]["PGA"]["categories"].index(
                            "Outstanding Producer of Theatrical Motion Pictures"
                        )
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Nowin_PGA"] = df["Win_PGA"].astype(bool).apply(lambda x: int(not x))

    # Critics Choice
    df["Nom_Criticschoice"] = df["Film"].map(
        {
            movie: (
                ("Critics Choice" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Picture"
                    in nominated_movies[movie]["awards"]["Critics Choice"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )
    df["Nonom_Criticschoice"] = (
        df["Nom_Criticschoice"].astype(bool).apply(lambda x: int(not x))
    )

    df["Win_Criticschoice"] = df["Film"].map(
        {
            movie: (
                ("Critics Choice" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Picture"
                    in nominated_movies[movie]["awards"]["Critics Choice"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["Critics Choice"]["results"][
                        nominated_movies[movie]["awards"]["Critics Choice"][
                            "categories"
                        ].index("Best Picture")
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )
    df["Nowin_Criticschoice"] = (
        df["Win_Criticschoice"].astype(bool).apply(lambda x: int(not x))
    )

    # Save

    df.to_csv(f"data/oscardata_{new_season}_bestpicture{suffix}.csv", index=False)


def create_newseason_acting_dataframe(
    nominated_movies, nominated_actors, new_season, suffix=""
):
    print(f"Creating {new_season} dataframe for acting")

    df = pd.read_excel(f"data/nominations_{new_season}.xlsx")
    df = df[
        df["Category"].apply(
            lambda x: ("actor" in x.lower()) or ("actress") in x.lower()
        )
    ]
    df["Film"] = df["Film"].astype(str)

    start_cols = df.columns
    final_cols = pd.read_csv(f"data/oscardata_acting.csv").columns
    for col in final_cols:
        if col not in start_cols:
            df[col] = np.nan

    # Rating columns
    df["Rating_IMDB"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["IMDB_rating"]
            for movie in nominated_movies.keys()
        }
    )
    df["Rating_rtcritic"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["RT_critics"]
            for movie in nominated_movies.keys()
        }
    )
    df["Rating_rtaudience"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["RT_audience"]
            for movie in nominated_movies.keys()
        }
    )

    # Oscar stat columns
    df["Oscarstat_totalnoms"] = df["Film"].map(
        {
            movie: len(nominated_movies[movie]["awards"]["Oscar"]["categories"])
            for movie in nominated_movies.keys()
        }
    )

    df["Oscarstat_previousnominations_acting"] = df["Nominee"].map(
        {
            actor: nominated_actors[actor]["oscar_nominations"] - 1
            for actor in nominated_actors.keys()
        }
    )

    df["Oscarstat_previouswins_acting"] = df["Nominee"].map(
        {
            actor: nominated_actors[actor]["oscar_wins"]
            for actor in nominated_actors.keys()
        }
    )

    df["Oscarstat_previousnominee_acting"] = (
        df["Oscarstat_previousnominations_acting"] > 0
    ) * 1

    df["Oscarstat_previouswinner_acting"] = (
        df["Oscarstat_previouswins_acting"] > 0
    ) * 1

    # Genre columns
    genre_cols = [col for col in final_cols if "Genre" in col]
    for genre in genre_cols:
        df[genre] = df["Film"].map(
            {
                movie: (genre.split("_")[1] in nominated_movies[movie]["genres"]) * 1
                for movie in nominated_movies.keys()
            }
        )

    # MPAA columns
    MPAA_rating_types = [
        col for col in final_cols if ("MPAA" in col) and ("rating" not in col)
    ]
    df["MPAA_rating"] = df["Film"].map(
        {movie: nominated_movies[movie]["MPAA"] for movie in nominated_movies.keys()}
    )
    df[MPAA_rating_types] = 0
    for rating_type in MPAA_rating_types:
        df.loc[df["MPAA_rating"] == rating_type.split("_")[1], rating_type] = 1

    # Release columns
    df["Release_date"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["release_date"]
            for movie in nominated_movies.keys()
        }
    )
    releaseQ_cols = [col for col in final_cols if "Release_Q" in col]
    for Q in releaseQ_cols:
        Q_num = int(Q[-1])
        df[Q] = df["Film"].map(
            {
                movie: (nominated_movies[movie]["release_quarter"] == Q_num) * 1
                for movie in nominated_movies.keys()
            }
        )

    # Age, birthyear and gender columns
    df["Birthyear"] = df["Nominee"].map(
        {
            actor: nominated_actors[actor]["birthyear"]
            for actor in nominated_actors.keys()
        }
    )
    df["Age"] = df["Nominee"].map(
        {actor: nominated_actors[actor]["age"] for actor in nominated_actors.keys()}
    )
    df["Female"] = df["Category"].isin(["Actress", "Supporting Actress"]) * 1

    age_cols = [col for col in final_cols if "Age_[" in col]
    for age_col in age_cols:
        age_bucket_bounds = list(map(int, re.findall("\d+", age_col))) + [np.inf]
        lb = age_bucket_bounds[0]
        ub = age_bucket_bounds[1]
        df[age_col] = df["Age"].apply(lambda x: (lb < x and x <= ub) * 1)

    # Nom and Win columns

    # Oscar
    df["Nom_Oscar_bestfilm"] = df["Film"].map(
        {
            movie: (
                "Best Motion Picture of the Year"
                in nominated_movies[movie]["awards"]["Oscar"]["categories"]
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    oscar_acting_cats = ["Actor", "Actress", "Supporting Actor", "Supporting Actress"]

    # BAFTA
    BAFTA_acting_cats = [
        "Best Leading Actor",
        "Best Leading Actress",
        "Best Supporting Actor",
        "Best Supporting Actress",
    ]
    for oscar_cat, bafta_cat in zip(oscar_acting_cats, BAFTA_acting_cats):
        df.loc[df["Category"] == oscar_cat, "Nom_BAFTA"] = df["Nominee"].map(
            {
                actor: (
                    ("BAFTA" in nominated_movies[movie]["awards"].keys())
                    and (
                        bafta_cat
                        in nominated_movies[movie]["awards"]["BAFTA"]["categories"]
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == oscar_cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

    for oscar_cat, bafta_cat in zip(oscar_acting_cats, BAFTA_acting_cats):
        df.loc[df["Category"] == oscar_cat, "Win_BAFTA"] = df["Nominee"].map(
            {
                actor: (
                    ("BAFTA" in nominated_movies[movie]["awards"].keys())
                    and (
                        bafta_cat
                        in nominated_movies[movie]["awards"]["BAFTA"]["categories"]
                    )
                    and (
                        nominated_movies[movie]["awards"]["BAFTA"]["results"][
                            nominated_movies[movie]["awards"]["BAFTA"][
                                "categories"
                            ].index(bafta_cat)
                        ]
                        == "Winner"
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == oscar_cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

    # Golden Globe
    df.loc[
        ~df["Category"].isin(["Actress", "Actor"]), "Win_GoldenGlobe_comedy-leadacting"
    ] = 0
    df.loc[
        ~df["Category"].isin(["Actress", "Actor"]), "Nom_GoldenGlobe_drama-leadacting"
    ] = 0
    df.loc[
        ~df["Category"].isin(["Actress", "Actor"]), "Win_GoldenGlobe_drama-leadacting"
    ] = 0
    df.loc[
        ~df["Category"].isin(["Actress", "Actor"]), "Nom_GoldenGlobe_comedy-leadacting"
    ] = 0
    df.loc[
        ~df["Category"].isin(["Supporting Actress", "Supporting Actress"]),
        "Win_GoldenGlobe_supportingacting",
    ] = 0
    df.loc[
        ~df["Category"].isin(["Supporting Actress", "Supporting Actress"]),
        "Nom_GoldenGlobe_supportingacting",
    ] = 0
    for cat in ["Actress", "Actor"]:
        df.loc[df["Category"] == cat, "Nom_GoldenGlobe_drama-leadacting"] = df.loc[
            df["Category"] == cat, "Nominee"
        ].map(
            {
                actor: (
                    ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                    and (
                        f"Best Performance by an {cat} in a Motion Picture - Drama"
                        in nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ]
                    )
                    and (
                        actor
                        in nominated_movies[movie]["awards"]["Golden Globe"]["to"][
                            nominated_movies[movie]["awards"]["Golden Globe"][
                                "categories"
                            ].index(
                                f"Best Performance by an {cat} in a Motion Picture - Drama"
                            )
                        ]
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

        df.loc[df["Category"] == cat, "Win_GoldenGlobe_drama-leadacting"] = df.loc[
            df["Category"] == cat, "Nominee"
        ].map(
            {
                actor: (
                    ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                    and (
                        f"Best Performance by an {cat} in a Motion Picture - Drama"
                        in nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ]
                    )
                    and (
                        actor
                        in nominated_movies[movie]["awards"]["Golden Globe"]["to"][
                            nominated_movies[movie]["awards"]["Golden Globe"][
                                "categories"
                            ].index(
                                f"Best Performance by an {cat} in a Motion Picture - Drama"
                            )
                        ]
                    )
                    and (
                        nominated_movies[movie]["awards"]["Golden Globe"]["results"][
                            nominated_movies[movie]["awards"]["Golden Globe"][
                                "categories"
                            ].index(
                                f"Best Performance by an {cat} in a Motion Picture - Drama"
                            )
                        ]
                        == "Winner"
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

        df.loc[df["Category"] == cat, "Nom_GoldenGlobe_comedy-leadacting"] = df.loc[
            df["Category"] == cat, "Nominee"
        ].map(
            {
                actor: (
                    ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                    and (
                        f"Best Performance by an {cat} in a Motion Picture - Musical or Comedy"
                        in nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ]
                    )
                    and (
                        actor
                        in nominated_movies[movie]["awards"]["Golden Globe"]["to"][
                            nominated_movies[movie]["awards"]["Golden Globe"][
                                "categories"
                            ].index(
                                f"Best Performance by an {cat} in a Motion Picture - Musical or Comedy"
                            )
                        ]
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

        df.loc[df["Category"] == cat, "Win_GoldenGlobe_comedy-leadacting"] = df.loc[
            df["Category"] == cat, "Nominee"
        ].map(
            {
                actor: (
                    ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                    and (
                        f"Best Performance by an {cat} in a Motion Picture - Musical or Comedy"
                        in nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ]
                    )
                    and (
                        actor
                        in nominated_movies[movie]["awards"]["Golden Globe"]["to"][
                            nominated_movies[movie]["awards"]["Golden Globe"][
                                "categories"
                            ].index(
                                f"Best Performance by an {cat} in a Motion Picture - Musical or Comedy"
                            )
                        ]
                    )
                    and (
                        nominated_movies[movie]["awards"]["Golden Globe"]["results"][
                            nominated_movies[movie]["awards"]["Golden Globe"][
                                "categories"
                            ].index(
                                f"Best Performance by an {cat} in a Motion Picture - Musical or Comedy"
                            )
                        ]
                        == "Winner"
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

        df.loc[
            df["Category"] == "Supporting " + cat, "Nom_GoldenGlobe_supportingacting"
        ] = df.loc[df["Category"] == "Supporting " + cat, "Nominee"].map(
            {
                actor: (
                    ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                    and (
                        f"Best Performance by an {cat} in a Supporting Role in a Motion Picture"
                        in nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ]
                    )
                    and (
                        actor
                        in [
                            a[0]
                            for a, c in zip(
                                nominated_movies[movie]["awards"]["Golden Globe"]["to"],
                                nominated_movies[movie]["awards"]["Golden Globe"][
                                    "categories"
                                ],
                            )
                            if c
                            == f"Best Performance by an {cat} in a Supporting Role in a Motion Picture"
                        ]
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == "Supporting " + cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

        df.loc[
            df["Category"] == "Supporting " + cat, "Win_GoldenGlobe_supportingacting"
        ] = df.loc[df["Category"] == "Supporting " + cat, "Nominee"].map(
            {
                actor: (
                    ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                    and (
                        f"Best Performance by an {cat} in a Supporting Role in a Motion Picture"
                        in nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ]
                    )
                    and (
                        actor
                        in [
                            a[0]
                            for a, c in zip(
                                nominated_movies[movie]["awards"]["Golden Globe"]["to"],
                                nominated_movies[movie]["awards"]["Golden Globe"][
                                    "categories"
                                ],
                            )
                            if c
                            == f"Best Performance by an {cat} in a Supporting Role in a Motion Picture"
                        ]
                    )
                    and (
                        [
                            r
                            for a, c, r in zip(
                                nominated_movies[movie]["awards"]["Golden Globe"]["to"],
                                nominated_movies[movie]["awards"]["Golden Globe"][
                                    "categories"
                                ],
                                nominated_movies[movie]["awards"]["Golden Globe"][
                                    "results"
                                ],
                            )
                            if (
                                c
                                == f"Best Performance by an {cat} in a Supporting Role in a Motion Picture"
                            )
                            and (a[0] == actor)
                        ][0]
                        == "Winner"
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == "Supporting " + cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

    # SAG
    SAG_acting_cats = [
        "Outstanding Performance by a Male Actor in a Leading Role",
        "Outstanding Performance by a Female Actor in a Leading Role",
        "Outstanding Performance by a Male Actor in a Supporting Role",
        "Outstanding Performance by a Female Actor in a Supporting Role",
    ]
    for oscar_cat, SAG_cat in zip(oscar_acting_cats, SAG_acting_cats):
        df.loc[df["Category"] == oscar_cat, "Nom_SAG_acting"] = df["Nominee"].map(
            {
                actor: (
                    ("SAG" in nominated_movies[movie]["awards"].keys())
                    and (
                        SAG_cat
                        in nominated_movies[movie]["awards"]["SAG"]["categories"]
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == oscar_cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

        df.loc[df["Category"] == oscar_cat, "Win_SAG_acting"] = df["Nominee"].map(
            {
                actor: (
                    ("SAG" in nominated_movies[movie]["awards"].keys())
                    and (
                        SAG_cat
                        in nominated_movies[movie]["awards"]["SAG"]["categories"]
                    )
                    and (
                        nominated_movies[movie]["awards"]["SAG"]["results"][
                            nominated_movies[movie]["awards"]["SAG"][
                                "categories"
                            ].index(SAG_cat)
                        ]
                        == "Winner"
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == oscar_cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

    df["Nom_SAG_bestcast"] = df["Film"].map(
        {
            movie: (
                ("SAG" in nominated_movies[movie]["awards"].keys())
                and (
                    "Outstanding Performance by a Cast in a Motion Picture"
                    in nominated_movies[movie]["awards"]["SAG"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Win_SAG_bestcast"] = df["Film"].map(
        {
            movie: (
                ("SAG" in nominated_movies[movie]["awards"].keys())
                and (
                    "Outstanding Performance by a Cast in a Motion Picture"
                    in nominated_movies[movie]["awards"]["SAG"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["SAG"]["results"][
                        nominated_movies[movie]["awards"]["SAG"]["categories"].index(
                            "Outstanding Performance by a Cast in a Motion Picture"
                        )
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Nowin_SAG_acting"] = (
        df["Win_SAG_acting"].astype(bool).apply(lambda x: int(not x))
    )
    df["Nonom_SAG_acting"] = (
        df["Nom_SAG_acting"].astype(bool).apply(lambda x: int(not x))
    )
    df["Nonom_SAG_bestcast"] = (
        df["Nom_SAG_bestcast"].astype(bool).apply(lambda x: int(not x))
    )
    df["Nowin_SAG_bestcast"] = (
        df["Win_SAG_bestcast"].astype(bool).apply(lambda x: int(not x))
    )

    # Critics Choice
    CC_acting_cats = [
        "Best Actor",
        "Best Actress",
        "Best Supporting Actor",
        "Best Supporting Actress",
    ]
    for oscar_cat, cc_cat in zip(oscar_acting_cats, CC_acting_cats):
        df.loc[df["Category"] == oscar_cat, "Nom_Criticschoice"] = df["Nominee"].map(
            {
                actor: (
                    ("Critics Choice" in nominated_movies[movie]["awards"].keys())
                    and (
                        cc_cat
                        in nominated_movies[movie]["awards"]["Critics Choice"][
                            "categories"
                        ]
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == oscar_cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

        df.loc[df["Category"] == oscar_cat, "Win_Criticschoice"] = df["Nominee"].map(
            {
                actor: (
                    ("Critics Choice" in nominated_movies[movie]["awards"].keys())
                    and (
                        cc_cat
                        in nominated_movies[movie]["awards"]["Critics Choice"][
                            "categories"
                        ]
                    )
                    and (
                        nominated_movies[movie]["awards"]["Critics Choice"]["results"][
                            nominated_movies[movie]["awards"]["Critics Choice"][
                                "categories"
                            ].index(cc_cat)
                        ]
                        == "Winner"
                    )
                )
                * 1
                for movie, actor in df[df["Category"] == oscar_cat][
                    ["Film", "Nominee"]
                ].values
            }
        )

    df["Nowin_Criticschoice"] = (
        df["Win_Criticschoice"].astype(bool).apply(lambda x: int(not x))
    )
    df["Nonom_Criticschoice"] = (
        df["Nom_Criticschoice"].astype(bool).apply(lambda x: int(not x))
    )

    # Save

    df.to_csv(f"data/oscardata_{new_season}_acting{suffix}.csv", index=False)


def create_newseason_director_dataframe(
    nominated_movies, nominated_directors, new_season, suffix=""
):
    print(f"Creating {new_season} dataframe for Best Director")

    df = pd.read_excel(f"data/nominations_{new_season}.xlsx")
    df = df[df["Category"] == "Director"]
    df["Film"] = df["Film"].astype(str)

    start_cols = df.columns
    final_cols = pd.read_csv(f"data/oscardata_bestdirector.csv").columns
    for col in final_cols:
        if col not in start_cols:
            df[col] = np.nan

    # Rating columns
    df["Rating_IMDB"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["IMDB_rating"]
            for movie in nominated_movies.keys()
        }
    )
    df["Rating_rtcritic"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["RT_critics"]
            for movie in nominated_movies.keys()
        }
    )
    df["Rating_rtaudience"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["RT_audience"]
            for movie in nominated_movies.keys()
        }
    )

    # Oscar stat columns
    df["Oscarstat_totalnoms"] = df["Film"].map(
        {
            movie: len(nominated_movies[movie]["awards"]["Oscar"]["categories"])
            for movie in nominated_movies.keys()
        }
    )

    df["Oscarstat_previousnominations_bestdirector"] = df["Nominee"].map(
        {
            director: nominated_directors[director]["oscar_nominations"] - 1
            for director in nominated_directors.keys()
        }
    )

    df["Oscarstat_previouswins_bestdirector"] = df["Nominee"].map(
        {
            director: nominated_directors[director]["oscar_wins"]
            for director in nominated_directors.keys()
        }
    )

    # Genre columns
    genre_cols = [col for col in final_cols if "Genre" in col]
    for genre in genre_cols:
        df[genre] = df["Film"].map(
            {
                movie: (genre.split("_")[1] in nominated_movies[movie]["genres"]) * 1
                for movie in nominated_movies.keys()
            }
        )

    # MPAA columns
    MPAA_rating_types = [
        col for col in final_cols if ("MPAA" in col) and ("rating" not in col)
    ]
    df["MPAA_rating"] = df["Film"].map(
        {movie: nominated_movies[movie]["MPAA"] for movie in nominated_movies.keys()}
    )
    df[MPAA_rating_types] = 0
    for rating_type in MPAA_rating_types:
        df.loc[df["MPAA_rating"] == rating_type.split("_")[1], rating_type] = 1

    # Release columns
    df["Release_date"] = df["Film"].map(
        {
            movie: nominated_movies[movie]["release_date"]
            for movie in nominated_movies.keys()
        }
    )
    releaseQ_cols = [col for col in final_cols if "Release_Q" in col]
    for Q in releaseQ_cols:
        Q_num = int(Q[-1])
        df[Q] = df["Film"].map(
            {
                movie: (nominated_movies[movie]["release_quarter"] == Q_num) * 1
                for movie in nominated_movies.keys()
            }
        )

    # Award columns

    # Oscar
    df["Nom_Oscar_bestfilm"] = df["Film"].map(
        {
            movie: (
                "Best Motion Picture of the Year"
                in nominated_movies[movie]["awards"]["Oscar"]["categories"]
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    # DGA
    df["Nom_DGA"] = df["Film"].map(
        {
            movie: ("DGA" in nominated_movies[movie]["awards"].keys()) * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Win_DGA"] = df["Film"].map(
        {
            movie: (
                ("DGA" in nominated_movies[movie]["awards"].keys())
                and (nominated_movies[movie]["awards"]["DGA"]["results"][0] == "Winner")
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    # Golden Globe
    df["Nom_GoldenGlobe_bestdirector"] = df["Film"].map(
        {
            movie: (
                ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Director - Motion Picture"
                    in nominated_movies[movie]["awards"]["Golden Globe"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Win_GoldenGlobe_bestdirector"] = df["Film"].map(
        {
            movie: (
                ("Golden Globe" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Director - Motion Picture"
                    in nominated_movies[movie]["awards"]["Golden Globe"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["Golden Globe"]["results"][
                        nominated_movies[movie]["awards"]["Golden Globe"][
                            "categories"
                        ].index("Best Director - Motion Picture")
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    # BAFTA
    df["Nom_BAFTA"] = df["Film"].map(
        {
            movie: (
                ("BAFTA" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Director"
                    in nominated_movies[movie]["awards"]["BAFTA"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    df["Win_BAFTA"] = df["Film"].map(
        {
            movie: (
                ("BAFTA" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Director"
                    in nominated_movies[movie]["awards"]["BAFTA"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["BAFTA"]["results"][
                        nominated_movies[movie]["awards"]["BAFTA"]["categories"].index(
                            "Best Director"
                        )
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )

    # Critics choice
    df["Nom_Criticschoice"] = df["Film"].map(
        {
            movie: (
                ("Critics Choice" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Director"
                    in nominated_movies[movie]["awards"]["Critics Choice"]["categories"]
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )
    df["Nonom_Criticschoice"] = (
        df["Nom_Criticschoice"].astype(bool).apply(lambda x: int(not x))
    )

    df["Win_Criticschoice"] = df["Film"].map(
        {
            movie: (
                ("Critics Choice" in nominated_movies[movie]["awards"].keys())
                and (
                    "Best Director"
                    in nominated_movies[movie]["awards"]["Critics Choice"]["categories"]
                )
                and (
                    nominated_movies[movie]["awards"]["Critics Choice"]["results"][
                        nominated_movies[movie]["awards"]["Critics Choice"][
                            "categories"
                        ].index("Best Director")
                    ]
                    == "Winner"
                )
            )
            * 1
            for movie in nominated_movies.keys()
        }
    )
    df["Nowin_Criticschoice"] = (
        df["Win_Criticschoice"].astype(bool).apply(lambda x: int(not x))
    )

    # Save

    df.to_csv(f"data/oscardata_{new_season}_bestdirector{suffix}.csv", index=False)


""" Main run function """


def run(new_season="2023"):
    # Load new season nominations
    df = pd.read_excel(f"data/nominations_{new_season}.xlsx")
    # Get (scrape) data
    nominated_movies, nominated_actors, nominated_directors = get_all_newseason_data(
        new_season=new_season
    )
    # Create picture dataframe for new season
    create_newseason_picture_dataframe(
        nominated_movies, new_season=new_season, suffix="-auto"
    )
    # Create acting dataframe for new season
    create_newseason_acting_dataframe(
        nominated_movies, nominated_actors, new_season=new_season, suffix="-auto"
    )
    # Create director dataframe for new season
    create_newseason_director_dataframe(
        nominated_movies, nominated_directors, new_season=new_season, suffix="-auto"
    )


run("2023")
