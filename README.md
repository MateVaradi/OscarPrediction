# OscarPrediction

This is (part of) the Python code and data for my BA thesis, _Predicting the Oscars with machine learning_.

## About the project
I use Logistic Regression and Random Forests to predict the Oscar winners of 6 Academy Award categories (Best Picture, Best Director, Best Actor in a Leading Role, Best Actress in a Leading Role, Best Actor in a Supporting Role, Best Actress in a Supporting Role). 

### Data
The dataset is built from the [Kaggle Oscar dataset](https://www.kaggle.com/unanimad/the-oscar-award). I scraped IMDB and Rotten Tomatoes to gather additional information about movies and awards. 

The variables used for prediction can be grouped into distinct categories:

- Film data: Genre, MPAA rating, IMDB score, Rotten Tomatoes Critics and Audience scores
- Oscar statistics: Total number of nominations, Number of previous nominations and wins (for actors and directors)
- Award data: the nominations and results of Oscar precursor award ceremonies. The following table gives a summary of these awards:

|                                  | Best Picture                                                         | Best Director                           | Best Lead Actor & Actress                                                                               | Best Supporting Actor & Actress                                                      |
|----------------------------------|----------------------------------------------------------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| BAFTA                            | Best Picture                                                         | Best Director                           | Best Lead Actor/Actress                                                                                 | Best Supporting Actor/Actress                                                        |
| Golden Globe                     | Best Motion Picture - Drama, Best Motion Picture - Musical or Comedy | Best Director                           | Best Performance in a Motion Picture – Drama,  Best Performance in a Motion Picture – Musical or Comedy | Best Supporting Actor/Actress                                                        |
| Critics Choice Award             | Best Picture                                                         | Best Director                           | Best Lead Actor/Actress                                                                                 | Best Supporting Actor/Actress                                                        |
| Screen Actors Guild Award        | Outstanding Performance by a Cast in a Motion Picture                |                                         | Outstanding Performance by a Cast in a Motion Picture                                                   | Best Supporting Actor/Actress, Outstanding Performance by a Cast in a Motion Picture |
| Directors Guild of America Award |                                                                      | Outstanding Achievement in Feature Film |                                                                                                         |                                                                                      |
| Producers Guild of America Award | Best Theatrical Motion Picture                                       |                                         |                                                                                                         |                                                                                      |
## Usage

### Prerequisites

- Standard data science packages (numpy, pandas, sklearn, etc)
- Web scraping packages (bs4, requests)
- [imdb](https://imdbpy.github.io/)
- [rotten_tomatoes_scraper](https://pypi.org/project/rotten-tomatoes-scraper/)

### Files in the directory

data: Contains the data used for the project
`example.ipynb`: Contains an example run to create predictions for the 2021 season
`new_season_data_scrape.py` Scrapes IMDB and Rotten Tomatoes to get data for the nominees of a new season
`predictor_selection.csv` Is a reference table which can be used to select the variables to include in the models.
`models.py` Contains all the functionality related to the machine learning models used in this project
`new_season_data_merge.py` Merges the _old_ dataset with the newly scraped data.

## References

- Váradi, Máté (2018) _Predicting the Oscars with machine learning._ Outstanding Student Paper, BCE, Statisztika és ökonometriai szekció.
