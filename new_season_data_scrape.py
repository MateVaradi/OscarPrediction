# Imports
from rotten_tomatoes_scraper.rt_scraper import MovieScraper
from imdb import IMDb
import pandas as pd
import numpy as np
import re
from time import sleep
from random import randint
from requests import get
import requests
from bs4 import BeautifulSoup
from requests import TooManyRedirects
from datetime import datetime

# Settings
pd.set_option('display.max_columns', 100)

""" Helper functions """


def quarter(date):
    month = date.split(' ')[1]
    if month in ['Jan', 'Feb', 'Mar']:
        q = 1
    elif month in ['Apr', 'May', 'Jun']:
        q = 2
    elif month in ['Jul', 'Aug', 'Sep']:
        q = 3
    elif month in ['Oct', 'Nov', 'Dec']:
        q = 4
    else:
        print('wrong date format')
    return q


""" Main functions """


def get_RT_ratings(movie_title):
    """
    Returns the Rotten Tomatoes critic score and audience score of a title
    """

    # Extract URL
    RT_search = MovieScraper()
    search_res = RT_search.search(movie_title)

    # Exact match
    url_list = [movie_dict['url'] for movie_dict in search_res['movies']
                if movie_dict['name'].lower() == movie_title.lower()]
    if len(url_list) == 1:
        url = url_list[0]
    # No exact match -  return the latest one
    elif not url_list:
        url_list = sorted([(movie_dict['url'], movie_dict['year']) for movie_dict in search_res['movies']],
                          key=lambda x: x[1], reverse=True)
        url = url_list[0][0]
        print(f'No exact match found. Going with {url}')
    # More than one exact match - return the latest one
    elif len(url_list) > 1:
        url_list = sorted([(movie_dict['url'], movie_dict['year']) for movie_dict in search_res['movies']
                           if movie_dict['name'].lower() == movie_title.lower()],
                          key=lambda x: x[1], reverse=True)
        url = url_list[0][0]
        print(f'More than one exact match found. Going with {url}')

    movie_scraper = MovieScraper(movie_url='https://www.rottentomatoes.com' + url)
    movie_scraper.extract_metadata()
    rt_critics_score = int(movie_scraper.metadata['Score_Rotten'])
    rt_audience_score = int(movie_scraper.metadata['Score_Audience'])
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
    ia = IMDb()
    res = ia._search_movie(movie_title, results=True)
    movie_ID = res[0][0]
    if res[0][1]['title'] != movie_title:
        print('Titles do not exactly match: ', movie_title, res[0][1]['title'])
    movie = ia.get_movie(movie_ID)
    rating = movie.data['rating']
    release_date = movie.data['original air date'].split(" (")[0]
    release_quarter = quarter(release_date)
    genres = [genre.lower() for genre in movie.data['genres']]
    MPAA = [c.split(':')[1] for c in movie.data['certificates'] if (c.startswith('United States') or 'USA' in c)][0]
    awards = dict()
    award_data = ia.get_movie_awards(movie_ID)['data']['awards']
    for nom in award_data:

        if nom['award'] == 'PGA Award':
            if 'PGA' not in awards.keys():
                awards['PGA'] = {'categories': [], 'results': [], 'to': []}
            if 'notes' in nom.keys():
                awards['PGA']['categories'].append(nom['notes'])
            else:
                awards['PGA']['categories'].append(nom['category'])
            awards['PGA']['results'].append(nom['result'])
            if 'to' in nom.keys():
                awards['PGA']['to'].append([str(n) for n in nom['to']])
            else:
                awards['PGA']['to'].append([])

        elif nom['award'] == 'BAFTA Film Award':
            if 'BAFTA' not in awards.keys():
                awards['BAFTA'] = {'categories': [], 'results': [], 'to': []}
            if 'notes' in nom.keys():
                awards['BAFTA']['categories'].append(nom['notes'])
            else:
                awards['BAFTA']['categories'].append(nom['category'])
            awards['BAFTA']['results'].append(nom['result'])
            if 'to' in nom.keys():
                awards['BAFTA']['to'].append([str(n) for n in nom['to']])
            else:
                awards['BAFTA']['to'].append([])

        elif nom['award'] == 'Golden Globe':
            if 'Golden Globe' not in awards.keys():
                awards['Golden Globe'] = {'categories': [], 'results': [], 'to': []}
            if 'notes' in nom.keys():
                awards['Golden Globe']['categories'].append(nom['notes'])
            else:
                awards['Golden Globe']['categories'].append(nom['category'])
            awards['Golden Globe']['results'].append(nom['result'])
            if 'to' in nom.keys():
                awards['Golden Globe']['to'].append([str(n) for n in nom['to']])
            else:
                awards['Golden Globe']['to'].append([])

        elif nom['award'] in ['Academy Awards, USA', 'Oscar']:
            if 'Oscar' not in awards.keys():
                awards['Oscar'] = {'categories': [], 'results': [], 'to': []}
            if 'notes' in nom.keys():
                awards['Oscar']['categories'].append(nom['notes'])
            else:
                awards['Oscar']['categories'].append(nom['category'])
            awards['Oscar']['results'].append(nom['result'])
            if 'to' in nom.keys():
                awards['Oscar']['to'].append([str(n) for n in nom['to']])
            else:
                awards['Oscar']['to'].append([])

        elif nom['award'] == 'Critics Choice Award':
            if 'Critics Choice' not in awards.keys():
                awards['Critics Choice'] = {'categories': [], 'results': [], 'to': []}
            if 'notes' in nom.keys():
                awards['Critics Choice']['categories'].append(nom['notes'])
            else:
                awards['Critics Choice']['categories'].append(nom['category'])
            awards['Critics Choice']['results'].append(nom['result'])
            if 'to' in nom.keys():
                awards['Critics Choice']['to'].append([str(n) for n in nom['to']])
            else:
                awards['Critics Choice']['to'].append([])

        elif nom['award'] == 'DGA Award':
            if 'DGA' not in awards.keys():
                awards['DGA'] = {'categories': [], 'results': [], 'to': []}
            if 'notes' in nom.keys():
                awards['DGA']['categories'].append(nom['notes'])
            else:
                awards['DGA']['categories'].append(nom['category'])
            awards['DGA']['results'].append(nom['result'])
            if 'to' in nom.keys():
                awards['DGA']['to'].append([str(n) for n in nom['to']])
            else:
                awards['DGA']['to'].append([])

        elif (nom['category'] == 'Screen Actors Guild Awards') or (nom['award'] == 'Screen Actors Guild Awards'):
            if 'SAG' not in awards.keys():
                awards['SAG'] = {'categories': [], 'results': [], 'to': []}
            if 'notes' in nom.keys():
                awards['SAG']['categories'].append(nom['notes'])
            else:
                awards['SAG']['categories'].append(nom['category'])
            awards['SAG']['results'].append(nom['result'])
            if 'to' in nom.keys():
                awards['SAG']['to'].append([str(n) for n in nom['to']])
            else:
                awards['SAG']['to'].append([])

    return rating, release_date, release_quarter, genres, MPAA, awards


def get_actor_data(actor_name):
    # Get actor data from IMDB API
    ia = IMDb()
    res = ia._search_person(actor_name, results=True)
    person_ID = res[0][0]
    found_name = ' '.join(res[0][1]['name'].split(',')[::-1]).strip(' ')
    if found_name.lower() != actor_name.lower():
        print('Titles do not exactly match: ', actor_name, found_name)
    person = ia.get_person(person_ID)
    birthyear = int(person.data['birth date'].split('-')[0])
    age = datetime.today().year - birthyear
    award_data = ia.get_person_awards(person_ID)['data']['awards']
    oscar_nominations = oscar_wins = 0
    for nom in award_data:
        if (nom['award'] == 'Academy Awards, USA') & \
                (nom['category'] in ['Best Performance by an Actress in a Leading Role',
                                     'Best Performance by an Actress in a Supporting Role',
                                     'Best Performance by an Actor in a Leading Role',
                                     'Best Performance by an Actor in a Supporting Role']):
            oscar_nominations += 1
            if 'result' in nom.keys():
                if nom['result'] == 'Winner':
                    oscar_wins += 1
    oscar_nominee = (oscar_nominations > 0) * 1
    oscar_winner = (oscar_wins > 0) * 1

    return age, birthyear, oscar_nominations, oscar_nominee, oscar_wins, oscar_winner


def get_director_data(director_name):
    # Get director data from IMDB API
    ia = IMDb()
    res = ia._search_person(director_name, results=True)
    person_ID = res[0][0]
    found_name = ' '.join(res[0][1]['name'].split(',')[::-1]).strip(' ')
    if found_name.lower() != director_name.lower():
        print('Titles do not exactly match: ', director_name, found_name)
    award_data = ia.get_person_awards(person_ID)['data']['awards']
    oscar_nominations = oscar_wins = 0
    for nom in award_data:
        if (nom['award'] == 'Academy Awards, USA') & (nom['category'] in
                                                      ['Best Director', 'Best Achievement in Directing']):
            oscar_nominations += 1
            if 'result' in nom.keys():
                if nom['result'] == 'Winner':
                    oscar_wins += 1
    oscar_nominee = (oscar_nominations > 0) * 1
    oscar_winner = (oscar_wins > 0) * 1

    return oscar_nominations, oscar_nominee, oscar_wins, oscar_winner


def get_all_movie_data(titles):
    new_movie_data = {}
    for movie_title in titles:
        movie_title = str(movie_title)
        print(f'Getting data for {movie_title}')
        new_movie_data[movie_title] = {}

        # Get data from IMDB
        tmp = get_IMDB_movie_data(movie_title)
        new_movie_data[movie_title]['IMDB_rating'] = tmp[0]
        new_movie_data[movie_title]['release_date'] = tmp[1]
        new_movie_data[movie_title]['release_quarter'] = tmp[2]
        new_movie_data[movie_title]['genres'] = tmp[3]
        new_movie_data[movie_title]['MPAA'] = tmp[4]
        new_movie_data[movie_title]['awards'] = tmp[5]
        # Get data from RT

        # sometimes the scrapers may fail and we need to do some hardcoded changes to make it work, for example
        if movie_title == 'Parasite':
            movie_title_RT = 'Parasite (Gisaengchung)'
        else:
            movie_title_RT = movie_title
        tmp = get_RT_ratings(movie_title_RT)
        new_movie_data[movie_title]['RT_critics'] = tmp[0]
        new_movie_data[movie_title]['RT_audience'] = tmp[1]

    return new_movie_data


def get_all_actor_data(names):
    new_actor_data = dict()
    for name in names:
        print(f'Getting data for {name}')
        new_actor_data[name] = {}
        tmp = get_actor_data(name)
        new_actor_data[name]['age'] = tmp[0]
        new_actor_data[name]['birthyear'] = tmp[1]
        new_actor_data[name]['oscar_nominations'] = tmp[2]
        new_actor_data[name]['oscar_nominee'] = tmp[3]
        new_actor_data[name]['oscar_wins'] = tmp[4]
        new_actor_data[name]['oscar_winner'] = tmp[5]

    return new_actor_data


def get_all_director_data(names):
    new_director_data = dict()
    for name in names:
        print(f'Getting data for {name}')
        new_director_data[name] = {}
        tmp = get_director_data(name)
        new_director_data[name]['oscar_nominations'] = tmp[0]
        new_director_data[name]['oscar_nominee'] = tmp[1]
        new_director_data[name]['oscar_wins'] = tmp[2]
        new_director_data[name]['oscar_winner'] = tmp[3]

    return new_director_data


def get_all_newseason_data(new_season):
    df = pd.read_excel(f'data/nominations {new_season}.xlsx')

    titles = df['Film'].unique()
    actors = df[df['Category'].apply(lambda x: ('actor' in x.lower()) or ('actress') in x.lower())]['Nominee'].unique()
    directors = df[df['Category'] == 'Director']['Nominee'].unique()

    nominated_movies = get_all_movie_data(titles)
    nominated_actors = get_all_actor_data(actors)
    nominated_directors = get_all_director_data(directors)

    return nominated_movies, nominated_actors, nominated_directors


def create_newseason_picture_dataframe(nominated_movies, new_season):
    df = pd.read_excel(f'data/nominations {new_season}.xlsx')
    df = df[df['Category'] == 'Picture']
    df['Film'] = df['Film'].astype(str)

    start_cols = df.columns
    final_cols = pd.read_csv(f'data/oscardata_bestpicture.csv').columns
    for col in final_cols:
        if col not in start_cols:
            df[col] = np.nan

    # Rating columns
    df['Rating_IMDB'] = df['Film'].map({movie: nominated_movies[movie]['IMDB_rating']
                                        for movie in nominated_movies.keys()})
    df['Rating_rtcritic'] = df['Film'].map({movie: nominated_movies[movie]['RT_critics']
                                            for movie in nominated_movies.keys()})
    df['Rating_rtaudience'] = df['Film'].map({movie: nominated_movies[movie]['RT_audience']
                                              for movie in nominated_movies.keys()})

    # Oscar stat columns
    df['Oscarstat_totalnoms'] = df['Film'].map({movie: len(nominated_movies[movie]['awards']['Oscar']['categories'])
                                                for movie in nominated_movies.keys()})

    # Genre columns
    genre_cols = [col for col in final_cols if 'Genre' in col]
    for genre in genre_cols:
        df[genre] = df['Film'].map({movie: (genre.split('_')[1] in nominated_movies[movie]['genres']) * 1
                                    for movie in nominated_movies.keys()})

    # MPAA columns
    MPAA_rating_types = [col for col in final_cols if ('MPAA' in col) and ('rating' not in col)]
    df['MPAA_rating'] = df['Film'].map({movie: nominated_movies[movie]['MPAA'] for movie in nominated_movies.keys()})
    df[MPAA_rating_types] = 0
    for rating_type in MPAA_rating_types:
        df.loc[df['MPAA_rating'] == rating_type.split('_')[1], rating_type] = 1

    # Release columns
    df['Release_date'] = df['Film'].map({movie: nominated_movies[movie]['release_date']
                                         for movie in nominated_movies.keys()})
    releaseQ_cols = [col for col in final_cols if 'Release_Q' in col]
    for Q in releaseQ_cols:
        Q_num = int(Q[-1])
        df[Q] = df['Film'].map({movie: (nominated_movies[movie]['release_quarter'] == Q_num) * 1
                                for movie in nominated_movies.keys()})

    # Nom and Win columns
    # Oscar
    df['Nom_Oscar_bestdirector'] = df['Film'].map({movie:
                                                       any(dir_cat in nominated_movies[movie]['awards']['Oscar'][
                                                           'categories']
                                                           for dir_cat in
                                                           ['Best Director', 'Best Achievement in Directing']) * 1
                                                   for movie in nominated_movies.keys()})

    # DGA
    df['Nom_DGA'] = df['Film'].map({movie: ('DGA' in nominated_movies[movie]['awards'].keys()) * 1
                                    for movie in nominated_movies.keys()})

    df['Win_DGA'] = df['Film'].map({movie: (('DGA' in nominated_movies[movie]['awards'].keys()) and
                                            (nominated_movies[movie]['awards']['DGA']['results'][0] == 'Winner')) * 1
                                    for movie in nominated_movies.keys()})

    # BAFTA
    df['Nom_BAFTA'] = df['Film'].map({movie: (('BAFTA' in nominated_movies[movie]['awards'].keys()) and
                                              ('Best Film' in nominated_movies[movie]['awards']['BAFTA'][
                                                  'categories'])) * 1
                                      for movie in nominated_movies.keys()})

    df['Win_BAFTA'] = df['Film'].map({movie: (('BAFTA' in nominated_movies[movie]['awards'].keys()) and
                                              ('Best Film' in nominated_movies[movie]['awards']['BAFTA']['categories'])
                                              and (nominated_movies[movie]['awards']['BAFTA']['categories'].index(
                'Best Film'))
                                              and (nominated_movies[movie]['awards']['BAFTA']['results'][
                                                       nominated_movies[movie]['awards']['BAFTA']['categories'].index(
                                                           'Best Film')] == 'Winner')) * 1
                                      for movie in nominated_movies.keys()})

    # Golden Globe
    df['Nom_GoldenGlobe_bestdrama'] = df['Film'].map(
        {movie: (('Golden Globe' in nominated_movies[movie]['awards'].keys()) and
                 ('Best Motion Picture - Drama' in nominated_movies[movie][
                     'awards']['Golden Globe']['categories'])) * 1
         for movie in nominated_movies.keys()})

    df['Nom_GoldenGlobe_bestcomedy'] = df['Film'].map(
        {movie: (('Golden Globe' in nominated_movies[movie]['awards'].keys()) and
                 ('Best Motion Picture - Musical or Comedy' in nominated_movies[movie][
                     'awards']['Golden Globe']['categories'])) * 1
         for movie in nominated_movies.keys()})

    df['Win_GoldenGlobe_bestdrama'] = df['Film'].map(
        {movie: (('Golden Globe' in nominated_movies[movie]['awards'].keys()) and
                 ('Best Motion Picture - Drama' in nominated_movies[movie][
                     'awards']['Golden Globe']['categories']) and
                 (nominated_movies[movie]['awards']['Golden Globe']['results'][
                      nominated_movies[movie]['awards']['Golden Globe']['categories'].index(
                          'Best Motion Picture - Drama')] == 'Winner')) * 1
         for movie in nominated_movies.keys()})

    df['Win_GoldenGlobe_bestcomedy'] = df['Film'].map(
        {movie: (('Golden Globe' in nominated_movies[movie]['awards'].keys()) and
                 ('Best Motion Picture - Musical or Comedy' in nominated_movies[movie][
                     'awards']['Golden Globe']['categories']) and
                 (nominated_movies[movie]['awards']['Golden Globe']['results'][
                      nominated_movies[movie]['awards']['Golden Globe']['categories'].index(
                          'Best Motion Picture - Musical or Comedy')] == 'Winner')
                 ) * 1 for movie in nominated_movies.keys()})

    # SAG
    df['Nom_SAG_acting'] = df['Film'].map({movie: (('SAG' in nominated_movies[movie]['awards'].keys()) and
                                                   ('Outstanding Performance by a Cast in a Motion Picture' in
                                                    nominated_movies[movie]['awards']['SAG']['categories'])) * 1
                                           for movie in nominated_movies.keys()})

    df['Nonom_SAG_acting'] = df['Nom_SAG_acting'].astype(bool).apply(lambda x: int(not x))

    df['Win_SAG_acting'] = df['Film'].map({movie: (('SAG' in nominated_movies[movie]['awards'].keys()) and
                                                   ('Outstanding Performance by a Cast in a Motion Picture' in
                                                    nominated_movies[movie]['awards']['SAG']['categories']) and
                                                   (nominated_movies[movie]['awards']['SAG']['results'][
                                                        nominated_movies[movie]['awards']['SAG']['categories'].index(
                                                            'Outstanding Performance by a Cast in a Motion Picture')]
                                                    == 'Winner')) * 1
                                           for movie in nominated_movies.keys()})

    df['Nowin_SAG_acting'] = df['Win_SAG_acting'].astype(bool).apply(lambda x: int(not x))

    # PGA
    df['Nom_PGA'] = df['Film'].map({movie: (('PGA' in nominated_movies[movie]['awards'].keys()) and
                                            ('Outstanding Producer of Theatrical Motion Pictures' in
                                             nominated_movies[movie]['awards']['PGA']['categories'])) * 1
                                    for movie in nominated_movies.keys()})

    df['Nonom_PGA'] = df['Nom_PGA'].astype(bool).apply(lambda x: int(not x))

    df['Win_PGA'] = df['Film'].map({movie: (('PGA' in nominated_movies[movie]['awards'].keys()) and
                                            ('Outstanding Producer of Theatrical Motion Pictures' in
                                             nominated_movies[movie]['awards']['PGA']['categories']) and
                                            (nominated_movies[movie]['awards']['PGA']['results'][
                                                 nominated_movies[movie]['awards']['PGA']['categories'].index(
                                                     'Outstanding Producer of Theatrical Motion Pictures')]
                                             == 'Winner')) * 1
                                    for movie in nominated_movies.keys()})

    df['Nowin_PGA'] = df['Win_PGA'].astype(bool).apply(lambda x: int(not x))

    # Critics Choice
    df['Nom_Criticschoice'] = df['Film'].map({movie: (('Critics Choice' in nominated_movies[movie]['awards'].keys()) and
                                                      ('Best Picture' in
                                                       nominated_movies[movie]['awards']['Critics Choice'][
                                                           'categories'])) * 1
                                              for movie in nominated_movies.keys()})
    df['Nonom_Criticschoice'] = df['Nom_Criticschoice'].astype(bool).apply(lambda x: int(not x))

    df['Win_Criticschoice'] = df['Film'].map({movie: (('Critics Choice' in nominated_movies[movie]['awards'].keys()) and
                                                      ('Best Picture' in
                                                       nominated_movies[movie]['awards']['Critics Choice'][
                                                           'categories']) and
                                                      (nominated_movies[movie]['awards']['Critics Choice']['results'][
                                                           nominated_movies[movie]['awards']['Critics Choice'][
                                                               'categories'].index(
                                                               'Best Picture')]
                                                       == 'Winner')) * 1 for movie in nominated_movies.keys()})
    df['Nowin_Criticschoice'] = df['Win_Criticschoice'].astype(bool).apply(lambda x: int(not x))

    # Save

    df.to_csv(f'data/TRY_oscardata_{new_season}_bestpicture.csv', index=False)


def create_newseason_acting_dataframe(nominated_movies, nominated_actors, new_season):
    print('wow')


def create_newseason_director_dataframe(nominated_movies, nominated_directors, new_season):
    print('wow')


""" Main run function """


def run(new_season='2020'):
    # Load new season nominations
    df = pd.read_excel(f'data/nominations {new_season}.xlsx')
    # Get (scrape) data
    nominated_movies, nominated_actors, nominated_directors = get_all_newseason_data(new_season=new_season)
    # Create picture dataframe for new season
    create_newseason_picture_dataframe(nominated_movies, new_season=new_season)
    # Create acting dataframe for new season
    # TODO     create acting dataframe
    create_newseason_acting_dataframe(nominated_movies, nominated_actors, new_season=new_season)
    # Create director dataframe for new season
    # TODO      create director dataframe
    create_newseason_director_dataframe(nominated_movies, nominated_directors, new_season=new_season)

# TODO   add awards to readme
# TODO after done:
#       - check against actual 2020 df
