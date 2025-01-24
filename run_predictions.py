import argparse

import numpy as np
import pandas as pd
import requests

from src.model import OscarPredictor
from src.utils import load_config
from src.visualizations import create_barcharts

config = load_config("configs/config.yml")

def process_json_output(response_json):
    res_df = pd.DataFrame(response_json)
    res_df["Winner"] = np.nan
    return res_df

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
        cat = model_category.lower().replace(" ","_")

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
        input_data = df_new[predictor.predictors+out_columns]

        # Prepare payload
        payload = {
            "data": input_data.to_dict(orient="records"),
            "vars": predictor.predictors
        }       

        # Call model
        # Send the POST request
        port = config["port_map"][model_category]
        response = requests.post(
            f"http://127.0.0.1:{port}/predict_{cat}",
            json=payload
        )

        print(response.json())

        # Save results
        category_res = process_json_output(response.json())
        all_res.append(category_res)

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
