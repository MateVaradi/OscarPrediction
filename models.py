"""
Runs models to obtain win predictions
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings

# Settings
pd.set_option('display.max_columns', 20)


def get_training_matrices(new_season='2020', data_folder='data/'):
    # Load which predictors to use
    variable_selection = pd.read_csv('predictor_selection.csv')
    picture_predictors = variable_selection['Variable'][variable_selection['Picture'].fillna(0) == 1]
    director_predictors = variable_selection['Variable'][variable_selection['Director'].fillna(0) == 1]
    lead_acting_predictors = variable_selection['Variable'][variable_selection['Lead Acting'].fillna(0) == 1]
    supporting_acting_predictors = variable_selection['Variable'][
        variable_selection['Supporting Acting'].fillna(0) == 1]

    # Load training data
    df_acting = pd.read_csv(data_folder + 'oscardata_acting.csv')
    df_picture = pd.read_csv(data_folder + 'oscardata_bestpicture.csv')
    df_director = pd.read_csv(data_folder + 'oscardata_bestdirector.csv')

    # Load data for new season to separate Dataframes
    df_acting_new = pd.read_csv(data_folder + 'oscardata_' + new_season + '_acting' + '.csv')
    df_picture_new = pd.read_csv(data_folder + 'oscardata_' + new_season + '_bestpicture' + '.csv')
    df_director_new = pd.read_csv(data_folder + 'oscardata_' + new_season + '_bestdirector' + '.csv')

    # Create training X,y matrices (store them in dictionaries)
    X_train = dict()
    y_train = dict()
    X_pred = dict()
    X_train['Picture'] = df_picture[picture_predictors]
    y_train['Picture'] = df_picture['Winner']
    X_train['Director'] = df_director[director_predictors]
    y_train['Director'] = df_director['Winner']
    for acting_category in ['Actor', 'Actress', 'Supporting Actor', 'Supporting Actress']:
        if 'Supporting' in acting_category:
            predictors = supporting_acting_predictors
        else:
            predictors = lead_acting_predictors

        X_train[acting_category] = df_acting.loc[df_acting['Category'] == acting_category, predictors]
        y_train[acting_category] = df_acting.loc[df_acting['Category'] == acting_category, 'Winner']

    # Create training matrices for new season
    X_pred['Picture'] = df_picture_new[picture_predictors]
    X_pred['Director'] = df_director_new[director_predictors]
    for acting_category in ['Actor', 'Actress', 'Supporting Actor', 'Supporting Actress']:
        if 'Supporting' in acting_category:
            predictors = supporting_acting_predictors
        else:
            predictors = lead_acting_predictors
        X_pred[acting_category] = df_acting_new.loc[df_acting_new['Category'] == acting_category, predictors]

    return X_train, y_train, X_pred


def predict_winners(model_type, new_season, new_season_df,
                    out_columns=['Category', 'Film', 'Nominee', 'Year', 'Prob', 'Classification']):
    """
    Makes predictions for the new season in all categories

    :param model_type: "logit" or "random forest" - which model to run
    :param season: season year (str)
    :param new_season_df: Dataframe containing information about nominees
    :param out_columns: list of columns to return in summary tables

    """
    # Load X and y matrices
    X_train, y_train, X_pred = get_training_matrices(new_season)

    # Get predictions for all categories
    prediction_dfs = []
    for category in new_season_df['Category'].unique():
        print(f'Creating predictions for {category}')
        if model_type.lower() in ['logit', 'logistic regression']:
            model = LogisticRegressionCV(max_iter=1000)
        elif model_type.lower() in ['rf', 'random forest']:
            model = RandomForestClassifier(n_estimators=250)
        else:
            warnings.warn(f'Model type {model_type} not supported')

        # Train model and obtain probabilities for new season
        X = X_train[category]
        y = y_train[category]
        X_t = X_pred[category]
        model.fit(X, y)
        category_df = new_season_df[new_season_df['Category'] == category].copy()
        probs = model.predict_proba(X_t)[:, 1]
        category_df['Prob'] = probs

        # Classify the film with the highest probability of winning as the winner
        maxprob = max(category_df['Prob'])
        category_df['Classification'] = (category_df['Prob'] == maxprob) * 1

        prediction_dfs.append(category_df[out_columns])

    prediction_df = pd.concat(prediction_dfs)
    winners = prediction_df[prediction_df['Classification'] == 1]

    return prediction_df, winners


new_season_nominees = pd.read_excel('data/nominations_2020.xlsx')
prediction_df, winners = predict_winners('rf', '2020', new_season_nominees)

# TODO
#  - logit convergence
#  - add awards to readme
# TODO ?
#  - RF categorical vars instead of dummies?
#  - try 2 acting models instead of 4
