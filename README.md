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
- Web scraping packages (`bs4`, `requests`, `selenium`, `webdriver`)
- [https://imdbpy.github.io/](https://pypi.org/project/imdbmovies/)
- [BentoML](https://github.com/bentoml/BentoML)

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

2.1 Simply run `python get_predictions.py --year <year>`. This will call the latest deployed models for each category (see the section, **Model maintenance & deployment** for details).


#### 3. Dataset maintenance

3.1 After the winners have been announced, add a *Winner* column in `oscardata_<year>_<category>.csv` for each category, marking the winners with a 1. 

3.2 Run `python merge_new_season_data.py --year <year>` to update the main datasets in `data/` with the results from the current year.

#### 4. Model maintenance & deployment

Models can be trained by running `train.py`. This will retrain a logistic regression and a random forest model by default for each category. The models will be saved to BentoML.

To deploy the latest trained model, a separate script must be run for each of the 4 model categories, eg.:
`bentoml serve deploy_supporting_acting.py --port 3003 &` will deploy the model used to predict the Supporting Acting categories. 

Each model should be deployed to a different port. The recommended port allocations can be found in `config/configs.yml`.

When a port is already in use, you can run

`lsof -i :<port>` to check the used process IDs (PID)

`kill <PID>`to kill the process

#### 5. Model development

You can play around with  `notebooks/model_development.ipynb` if you want to improve the models, try different hyperparameters, predictor sets, etc. Examples are provided in the notebook

## References

- Váradi, Máté (2018) _Predicting the Oscars with machine learning._ Outstanding Student Paper, BCE, Statisztika és ökonometriai szekció.
