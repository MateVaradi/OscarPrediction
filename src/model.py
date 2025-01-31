import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


class OscarPredictor:
    def __init__(
        self,
        model_category,
        config,
        new_season=None,
        model_type="logit",
        verbose=False,
    ):

        self.new_season = new_season
        self.cfg = config
        self.verbose = verbose

        if model_category in [
            "Picture",
            "Director",
            "Supporting Acting",
            "Lead Acting",
        ]:
            self.model_category = model_category
            self.oscar_categories = self.cfg["category_map"][self.model_category]
        else:
            raise ValueError(f"Category: {model_category} not recognized")
        if self.verbose:
            print(f"Setting up Predictor for {model_category}")

        self.model_type = model_type

        # Load which predictors to use
        variable_selection = pd.read_excel(
            self.cfg["predictor_dir"], sheet_name=self.cfg["predictor_set"]
        )
        self.predictors = variable_selection["Variable"][
            variable_selection[self.model_category].fillna(0) == 1
        ].values.tolist()

    def load_train_data(self):
        # Load data
        df_train = pd.read_csv(self.cfg["data_dirs"][self.model_category])
        df_train = df_train[df_train["Category"].isin(self.oscar_categories)]

        return df_train

    def load_new_data(self):
        new_data_dir = self.cfg["data_dirs"][self.model_category].replace(
            "_", f"_{self.new_season}_"
        )
        df_new = pd.read_csv(new_data_dir)
        df_new = df_new[df_new["Category"].isin(self.oscar_categories)]

        return df_new

    def define_model(self):
        if self.model_type.lower() in ["logit", "logistic regression"]:
            model = LogisticRegressionCV(max_iter=5000, solver="newton-cg")
        elif self.model_type.lower() in ["rf", "random forest"]:
            model = RandomForestClassifier(n_estimators=250)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

        return model

    def predict_new_season(self):
        if not hasattr(self, "model"):
            raise ValueError("Model needs to be trained first")

        df_new = self.load_new_data()
        X_pred = df_new[self.predictors]
        df_res = self.get_predictions(self.model, df_new)

        self.df_res_new = df_res

    def train_model_cv(self, model=None):
        print(f"Training model for {self.model_category}")

        if model is None:
            model = self.define_model()

        df_train = self.load_train_data()

        # Get CV metrics
        metrics = self.eval_cv(df_train, model)

        # Retrain on full data
        X = df_train[self.predictors]
        y = df_train["Winner"]
        model.fit(X, y)
        self.model = model
        df_res = self.get_predictions(model, df_train)

    def get_predictions(self, model, df_pred):

        df_res = df_pred.copy()
        X_pred = df_res[self.predictors]
        probs = model.predict_proba(X_pred)[:, 1]
        df_res["Prob"] = probs

        # Classify the film with the highest probability of winning as the winner
        df_res["Classification"] = 0
        win_idx = df_res.groupby(["Category", "Year"])["Prob"].idxmax()
        df_res.loc[win_idx, "Classification"] = 1

        out_columns = [
            "Category",
            "Film",
            "Nominee",
            "Year",
            "Winner",
            "Prob",
            "Classification",
        ]
        df_res = df_res[out_columns]

        return df_res

    def eval(self, df_res):
        auc_score = roc_auc_score(df_res["Winner"], df_res["Classification"])
        tpr_score = recall_score(df_res["Winner"], df_res["Classification"])
        print(f"AUC: {auc_score:.2f}, Recall: {tpr_score:.2f}")

    def eval_cv(self, df, model, n_splits=5):

        all_years = pd.Series(sorted(df["Year"].unique()))
        tscv = TimeSeriesSplit(test_size=5)

        auc_scores = []
        tpr_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(all_years)):
            train_years = all_years[train_idx].tolist()
            val_years = all_years[val_idx].tolist()
            df_train = df.loc[df["Year"].isin(train_years)]
            df_val = df.loc[df["Year"].isin(val_years)]
            X_train, X_val = df_train[self.predictors], df_val[self.predictors]
            y_train, y_val = df_train["Winner"], df_val["Winner"]

            # Train model
            model.fit(X_train, y_train)

            # Predict on validation data
            df_res = self.get_predictions(model, df_val)
            y_val_hat = df_res["Classification"]

            # Evaluate
            auc_score = roc_auc_score(y_val, y_val_hat)
            tpr_score = recall_score(y_val, y_val_hat)
            if self.verbose:
                print(f"AUC on fold {fold+1}: {auc_score:.2f}")
                print(f"Recall on fold {fold+1}: {tpr_score:.2f}")
            auc_scores.append(auc_score)
            tpr_scores.append(tpr_score)

        # Aggregate metrics
        avg_auc = np.mean(auc_scores)
        avg_tpr = np.mean(tpr_scores)
        metrics = {"AUC": avg_auc, "TPR": avg_tpr}
        print(
            f"Cross-Validated model results: AUC: {avg_auc:.2f}, Recall: {avg_tpr:.2f}"
        )

        return metrics


def process_results_df(df_pred, probs):
    df_res = df_pred.copy()

    df_res["Prob"] = probs

    # Classify the film with the highest probability of winning as the winner
    df_res["Classification"] = 0
    win_idx = df_res.groupby(["Category", "Year"])["Prob"].idxmax()
    df_res.loc[win_idx, "Classification"] = 1

    out_columns = [
        "Category",
        "Film",
        "Nominee",
        "Year",
        "Winner",
        "Prob",
        "Classification",
    ]
    df_res = df_res[out_columns]

    return df_res.to_dict(orient="records")
