import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split


class OscarPredictor:
    def __init__(
        self,
        new_season,
        model_category,
        config,
        model_type="logit",
        split_type="use_all",
    ):

        self.new_season = new_season
        self.cfg = config

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
        print(f"Setting up Predictor for {model_category}")

        self.split_type = split_type
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

    def prepare_train_data(self, df_train):
        X = df_train[self.predictors]
        y = df_train["Winner"]

        if self.split_type == "use_all":
            X_train = X
            y_train = y
            df_train = df_train
        else:
            raise ValueError(f"split_type : {self.split_type} not implemented")
            # TODO implement other splits

        return df_train, X_train, y_train

    def prepare_pred_data(self, df_new):
        X_pred = df_new[self.predictors]

        return X_pred

    def fit_model(self, X, y):
        if self.model_type.lower() in ["logit", "logistic regression"]:
            model = LogisticRegressionCV(max_iter=5000, solver="newton-cg")
        elif self.model_type.lower() in ["rf", "random forest"]:
            model = RandomForestClassifier(n_estimators=250)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

        model.fit(X, y)
        self.model = model

    def predict_new_season(self):
        if not hasattr(self, "model"):
            raise ValueError("Model needs to be trained first")

        df_new = self.load_new_data()
        X_pred = self.prepare_pred_data(df_new)
        df_res = self.get_predictions(X_pred, df_new)

        self.df_res_new = df_res

    def train_model(self):

        df_train = self.load_train_data()
        df_train, X_train, y_train = self.prepare_train_data(df_train)
        self.fit_model(X_train, y_train)
        df_res = self.get_predictions(X_train, df_train)

        # Evaluate results
        self.eval(df_res)
        self.df_res_train = df_res

    def get_predictions(self, X_pred, df_pred):

        df_res = df_pred.copy()
        probs = self.model.predict_proba(X_pred)[:, 1]
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
        print("AUC", roc_auc_score(df_res["Winner"], df_res["Classification"]))
        print("TPR", recall_score(df_res["Winner"], df_res["Classification"]))
