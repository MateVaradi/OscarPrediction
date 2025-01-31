# Oscar prediction

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

- Standard data science packages (`numpy`, `pandas`, `sklearn`, etc)
- Web scraping packages (`bs4`, `requests`)
- [imdb](https://imdbpy.github.io/)
- [rotten_tomatoes_scraper](https://pypi.org/project/rotten-tomatoes-scraper/)

### Files in the directory

- `data/`: Contains the data used for the project
- `config/`: Contains configurations files, such as
  - `predictor_selection.csv` Is a reference table which can be used to select the variables to include in the models.


### How to use

#### 1. Data scraping
1.1 Prepare the list of nominees for the given year. Use the existing `nominations_<year>.xlsx` files as a sample.

1.2 Prepare the scape config file - use `scrape_config_<year>.yml` as a sample, and add the relevant Wikipedia links of the year. If the layout of the Wikipedia tables changed since last year you might need to adjust the `maps` key of the yaml.

1.3 Run `python scrape_data.py --year <year>` to get data for the relevant Oscar season for <year> - run this a few days before the Oscar ceremony that you want to predict.

1.4 The resulting datasets will be called `oscardata_<year>_<category>-auto.csv`. Double check the dataset created and manually fill in any NaNs before running predictions. Once ready, you can save the changes to `oscardata_<year>_<category>.csv`.

#### 2. Predictions
2.1 Run `get_predictions.py`
2.2 Run `merge_new_season_data.py` to update your database with the actual winners after the Oscar ceremony.

'+ you can play around with  `model_development.py` if you want to improve the models, try different hyperparameters, predictor sets, etc. Examples are provided in the file

## References

- Váradi, Máté (2018) _Predicting the Oscars with machine learning._ Outstanding Student Paper, BCE, Statisztika és ökonometriai szekció.
