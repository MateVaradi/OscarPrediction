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

# Settings
pd.set_option('display.max_columns', 20)


def training_validation_split(year_selection, data_folder='data/', predictor_set='model_0'):
    """
    Returns training and validation X and y matrices (excluding data from the new season)

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

    # Further split training set into a training and a validation set (store them in dictionaries)
    X_train = dict()
    y_train = dict()
    X_val = dict()
    y_val = dict()
    X_train['Picture'] = df_picture.loc[~df_picture['Year'].isin(year_selection), picture_predictors]
    y_train['Picture'] = df_picture.loc[~df_picture['Year'].isin(year_selection), 'Winner']
    X_train['Director'] = df_director.loc[~df_director['Year'].isin(year_selection), director_predictors]
    y_train['Director'] = df_director.loc[~df_director['Year'].isin(year_selection), 'Winner']
    X_val['Picture'] = df_picture.loc[df_picture['Year'].isin(year_selection), picture_predictors]
    y_val['Picture'] = df_picture.loc[df_picture['Year'].isin(year_selection), ['Winner', 'Year', 'Category']]
    X_val['Director'] = df_director.loc[df_director['Year'].isin(year_selection), director_predictors]
    y_val['Director'] = df_director.loc[df_director['Year'].isin(year_selection), ['Winner', 'Year', 'Category']]
    for acting_category in ['Actor', 'Actress', 'Supporting Actor', 'Supporting Actress']:
        if 'Supporting' in acting_category:
            predictors = supporting_acting_predictors
        else:
            predictors = lead_acting_predictors

        X_train[acting_category] = df_acting.loc[(~df_acting['Year'].isin(year_selection)) &
                                                 (df_acting['Category'] == acting_category), predictors]
        y_train[acting_category] = df_acting.loc[(~df_acting['Year'].isin(year_selection)) &
                                                 (df_acting['Category'] == acting_category), 'Winner']
        X_val[acting_category] = df_acting.loc[(df_acting['Year'].isin(year_selection)) &
                                               (df_acting['Category'] == acting_category), predictors]
        y_val[acting_category] = df_acting.loc[(df_acting['Year'].isin(year_selection)) &
                                               (df_acting['Category'] == acting_category),
                                               ['Winner', 'Year', 'Category']]

    return X_train, y_train, X_val, y_val


def model_0_validation(model_type, year_selections, k=3, predictor_set='model_0'):
    """
    Makes predictions for the new season in all categories

    :param model_type: "logit" or "random forest" - which model to run
    :param season: season year (str)
    :param new_season_df: Dataframe containing information about nominees
    :param out_columns: list of columns to return in summary tables

    """
    # Prepare evaluation table
    out_table = pd.DataFrame(columns=['Category', 'AUC', 'TPR'])

    # Repeat validation k times:
    for i in range(k):
        year_selection = year_selections[i]
        # Load X and y matrices
        X_train, y_train, X_val, y_val = training_validation_split(year_selection, predictor_set=predictor_set)

        categories = X_train.keys()

        # Get predictions for all categories
        for category in categories:
            print(f'Training model for {category}')
            if model_type.lower() in ['logit', 'logistic regression']:
                model = LogisticRegressionCV(max_iter=5000,solver='newton-cg')
            elif model_type.lower() in ['rf', 'random forest']:
                model = RandomForestClassifier(n_estimators=250)
            else:
                warnings.warn(f'Model type {model_type} not supported')

            # Train model and obtain probabilities for new season
            X_t = X_train[category]
            y_t = y_train[category]
            X_v = X_val[category]
            val_years = y_val[category]['Year']
            model.fit(X_t, y_t)
            class_df = pd.DataFrame()
            class_df['Year'] = val_years
            probs = model.predict_proba(X_v)[:, 1]
            class_df['Prob'] = probs

            # Classify the film with the highest probability of winning as the winner
            class_df['Classification'] = 0
            for year in val_years.unique():
                maxprob = class_df.loc[class_df['Year'] == year, 'Prob'].max()
                class_df.loc[class_df['Year'] == year, 'Classification'] = \
                    (class_df.loc[class_df['Year'] == year, 'Prob'] == maxprob) * 1

            # Evaluate model and save results
            class_df['Winner'] = y_val[category]['Winner']
            out_table = out_table.append(
                {'Category': category,
                 'AUC': roc_auc_score(class_df['Winner'], class_df['Classification']),
                 'TPR': recall_score(class_df['Winner'], class_df['Classification'])}, ignore_index=True)

    out_table = out_table.groupby('Category').mean().reset_index()

    return out_table


def model_1_validation(model_type, year_selections, k=3, predictor_set='model_1'):
    """
    Makes predictions for the new season in all categories

    :param model_type: "logit" or "random forest" - which model to run
    :param season: season year (str)
    :param new_season_df: Dataframe containing information about nominees
    :param out_columns: list of columns to return in summary tables

    """
    # Prepare evaluation table
    out_table = pd.DataFrame(columns=['Category', 'AUC', 'TPR'])

    # Repeat validation k times:
    for i in range(k):
        # Load X and y matrices
        year_selection = year_selections[i]
        X_train, y_train, X_val, y_val = training_validation_split(year_selection, predictor_set=predictor_set)

        categories = {'Picture': ['Picture'],
                      'Director': ['Director'],
                      'Lead Acting': ['Actor', 'Actress'],
                      'Supporting Acting': ['Supporting Actor', 'Supporting Actress']}

        # Get predictions for all categories
        for model_cat, oscar_cats in categories.items():

            print(f'Training model for {model_cat}')
            if model_type.lower() in ['logit', 'logistic regression']:
                model = LogisticRegressionCV(max_iter=5000,solver='newton-cg')
            elif model_type.lower() in ['rf', 'random forest']:
                model = RandomForestClassifier(n_estimators=250)
            else:
                warnings.warn(f'Model type {model_type} not supported')

            # Train model and obtain probabilities for new season
            X_ts = []
            y_ts = []
            X_vs = []
            y_vs = []
            for oscar_cat in oscar_cats:
                X_ts.append(X_train[oscar_cat])
                y_ts.append(y_train[oscar_cat])
                X_vs.append(X_val[oscar_cat])
                y_vs.append(y_val[oscar_cat])

            X_t = pd.concat(X_ts)
            y_t = pd.concat(y_ts)
            X_v = pd.concat(X_vs)
            y_v = pd.concat(y_vs)

            model.fit(X_t, y_t)
            class_df = pd.DataFrame()
            val_years = y_v['Year']
            val_cats = y_v['Category']
            class_df['Year'] = val_years
            class_df['Category'] = val_cats
            probs = model.predict_proba(X_v)[:, 1]
            class_df['Prob'] = probs

            # Classify the film with the highest probability of winning as the winner
            class_df['Classification'] = 0
            for cat in val_cats:
                for year in val_years.unique():
                    maxprob = class_df.loc[(class_df['Year'] == year) & (class_df['Category'] == cat), 'Prob'].max()
                    class_df.loc[(class_df['Year'] == year) & (class_df['Category'] == cat), 'Classification'] = \
                        (class_df.loc[(class_df['Year'] == year) &
                                      (class_df['Category'] == cat), 'Prob'] == maxprob) * 1

            # Evaluate model and save results
            class_df['Winner'] = 0
            for cat in oscar_cats:
                class_df.loc[class_df['Category'] == cat, 'Winner'] = y_val[cat]['Winner']

                out_table = out_table.append(
                    {'Category': cat,
                     'AUC': roc_auc_score(class_df.loc[class_df['Category'] == cat, 'Winner'],
                                          class_df.loc[class_df['Category'] == cat, 'Classification']),
                     'TPR': recall_score(class_df.loc[class_df['Category'] == cat, 'Winner'],
                                         class_df.loc[class_df['Category'] == cat, 'Classification'])},
                    ignore_index=True)

    out_table = out_table.groupby('Category').mean().reset_index()

    return out_table


def model_3_validation(model_type, year_selections, k=3, predictor_set='model_0',
                       solver='lbfgs', tol=1e-4, Cs=10):
    """
    Makes predictions for the new season in all categories

    :param model_type: "logit" or "random forest" - which model to run
    :param season: season year (str)
    :param new_season_df: Dataframe containing information about nominees
    :param out_columns: list of columns to return in summary tables

    """
    # Prepare evaluation table
    out_table = pd.DataFrame(columns=['Category', 'AUC', 'TPR'])

    # Repeat validation k times:
    for i in range(k):
        year_selection = year_selections[i]
        # Load X and y matrices
        X_train, y_train, X_val, y_val = training_validation_split(year_selection, predictor_set=predictor_set)

        categories = X_train.keys()

        # Get predictions for all categories
        for category in categories:
            print(f'Training model for {category}')
            if model_type.lower() in ['logit', 'logistic regression']:
                model = LogisticRegressionCV(max_iter=5000, solver=solver, tol=tol, Cs=Cs)
            elif model_type.lower() in ['rf', 'random forest']:
                model = RandomForestClassifier(n_estimators=250)
            else:
                warnings.warn(f'Model type {model_type} not supported')

            # Train model and obtain probabilities for new season
            X_t = X_train[category]
            y_t = y_train[category]
            X_v = X_val[category]
            val_years = y_val[category]['Year']
            model.fit(X_t, y_t)
            class_df = pd.DataFrame()
            class_df['Year'] = val_years
            probs = model.predict_proba(X_v)[:, 1]
            class_df['Prob'] = probs

            # Classify the film with the highest probability of winning as the winner
            class_df['Classification'] = 0
            for year in val_years.unique():
                maxprob = class_df.loc[class_df['Year'] == year, 'Prob'].max()
                class_df.loc[class_df['Year'] == year, 'Classification'] = \
                    (class_df.loc[class_df['Year'] == year, 'Prob'] == maxprob) * 1

            # Evaluate model and save results
            class_df['Winner'] = y_val[category]['Winner']
            out_table = out_table.append(
                {'Category': category,
                 'AUC': roc_auc_score(class_df['Winner'], class_df['Classification']),
                 'TPR': recall_score(class_df['Winner'], class_df['Classification'])}, ignore_index=True)

    out_table = out_table.groupby('Category').mean().reset_index()

    return out_table


# Run different models

k = 5
start_year = 1961
end_year = 2019
num_seasons = end_year - start_year + 1
years = np.array(range(start_year, end_year + 1))
# later years have a larger change of being in the test set
propensity = np.power(years - 1950, 2) / sum(np.power(years - 1950, 2))
all_selected_years = np.random.choice(years, p=propensity, size=5 * round(num_seasons * 0.15), replace=False)
np.random.shuffle(all_selected_years)
year_selections = np.array_split(all_selected_years, 5)

# Model 0: the model used for the thesis
model_0_logit_out = model_0_validation(model_type='logit', k=k, year_selections=year_selections)
model_0_logit_out.to_csv('model_0_logit_metrics.csv', index=False)

model_0_rf_out = model_0_validation(model_type='rf', k=k, year_selections=year_selections)
model_0_rf_out.to_csv('model_0_rf_metrics.csv', index=False)

# Model 1: 2 acting categories instead of 4
model_1_logit_out = model_1_validation(model_type='logit', k=k, year_selections=year_selections)
model_1_logit_out.to_csv('model_1_logit_metrics.csv', index=False)

model_1_rf_out = model_1_validation(model_type='rf', k=k, year_selections=year_selections)
model_1_rf_out.to_csv('model_1_rf_metrics.csv', index=False)

# Model 2: categorical variables instead of dummies
model_2_rf_out = model_0_validation(model_type='rf', k=k, year_selections=year_selections, predictor_set='model_2')
model_2_rf_out.to_csv('model_2_rf_metrics.csv', index=False)

