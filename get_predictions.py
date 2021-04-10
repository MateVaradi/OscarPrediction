"""
Runs models to obtain win predictions
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

# Settings
pd.set_option('display.max_columns', 20)


def training_prediction_split(new_season='2020', data_folder='data/', predictor_set='model_1'):
    """
    Returns training X and y matrices and prediction test matrix for new season.

    :param new_season:
    :param data_folder:
    :return:
    """
    # Load which predictors to use
    variable_selection = pd.read_excel('predictor_selection.xlsx', sheet_name=predictor_set)
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

    # Create test X matrices for new season
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
    X_train, y_train, X_pred = training_prediction_split(new_season)

    # Define model categories
    categories = {'Picture': ['Picture'],
                  'Director': ['Director'],
                  'Lead Acting': ['Actor', 'Actress'],
                  'Supporting Acting': ['Supporting Actor', 'Supporting Actress']}

    # Get predictions for all categories
    prediction_dfs = []
    # Get predictions for all categories
    for model_cat, oscar_cats in categories.items():
        print(f'Training model for {model_cat}')
        if model_type.lower() in ['logit', 'logistic regression']:
            model = LogisticRegressionCV(max_iter=5000, solver='newton-cg')
        elif model_type.lower() in ['rf', 'random forest']:
            model = RandomForestClassifier(n_estimators=250)
        else:
            warnings.warn(f'Model type {model_type} not supported')

        # Get data
        Xs = []
        ys = []
        X_preds = []
        for oscar_cat in oscar_cats:
            Xs.append(X_train[oscar_cat])
            ys.append(y_train[oscar_cat])
            X_preds.append(X_pred[oscar_cat])

        X = pd.concat(Xs)
        y = pd.concat(ys)
        X_t = pd.concat(X_preds)

        # Train model and obtain probabilities for new season
        model.fit(X, y)
        category_df = new_season_df[new_season_df['Category'].isin(oscar_cats)].copy()
        probs = model.predict_proba(X_t)[:, 1]
        category_df['Prob'] = probs

        # Classify the film with the highest probability of winning as the winner
        for oscar_cat in oscar_cats:
            maxprob = max(category_df.loc[category_df['Category'] == oscar_cat, 'Prob'])
            category_df.loc[category_df['Category'] == oscar_cat, 'Classification'] = \
                (category_df.loc[category_df['Category'] == oscar_cat, 'Prob'] == maxprob) * 1

        prediction_dfs.append(category_df[out_columns])

    prediction_df = pd.concat(prediction_dfs)
    winners = prediction_df[prediction_df['Classification'] == 1]

    return prediction_df, winners

def run(new_season):
    """
    Create predictions for the new season
    """

    new_season_nominees = pd.read_excel(f'data/nominations_{new_season}.xlsx')
    prediction_df, winners = predict_winners('logit', '2020', new_season_nominees)

    # Save results
    save_loc = 'results/'
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    winners.to_csv(f'results/winner_predictions_{new_season}.csv', index=False)
    prediction_df.to_csv(f'results/all_predictions_{new_season}.csv', index=False)

run('2020')


