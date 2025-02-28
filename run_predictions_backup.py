import argparse

import numpy as np
import pandas as pd
import pickle

from src.model import OscarPredictor, process_results_df

from src.utils import load_config
from src.visualizations import create_barcharts

config = load_config("configs/config.yml")

def main():
    parser = argparse.ArgumentParser("run_predictions")
    parser.add_argument("--year", type=str, help="Year for which to run predictions")
    parser.add_argument(
        "--model_type",
        nargs="?",
        type=str,
        help="Which kind of model to use",
        default="rf",
    )
    args = parser.parse_args()

    all_res = []
    for model_category in config["category_map"].keys():
        cat_tag = model_category.lower().replace(" ","")

        predictor = OscarPredictor(
            model_category=model_category,
            model_type=args.model_type,
            new_season=args.year,
            config=config,
        )

        # Load prediction data
        df_new = predictor.load_new_data()
        out_columns = [
            "Category",
            "Film",
            "Nominee",
            "Year",
            "Winner"
        ]
        df_pred = df_new[predictor.predictors+out_columns]

    
        # Load saved sklearn model pickle
        model_name = f'{config["model_prefix"]}-{args.model_type}-{cat_tag}'
        with open(f"models/{model_name}.pkl", "rb") as f:
            model = pickle.load(f)

        # Call model
        result_df = model.get_predictions(model.model, df_pred)
        #result_df["Winner"] = np.nan

        # Save results
        all_res.append(result_df)

    prediction_df = pd.concat(all_res)
    res_dir = f"results/all_predictions_{args.model_type}_{args.year}.csv"
    prediction_df.to_csv(res_dir, index=False)

    # Save result barcharts
    create_barcharts(
        results_dir=res_dir,
        save_dir=f"results/predictions_barchart_{args.model_type}_{args.year}",
    )


if __name__ == "__main__":
    main()
