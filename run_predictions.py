import argparse

import pandas as pd

from src.model import OscarPredictor
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
        default="logit",
    )
    args = parser.parse_args()

    all_res = []
    for model_category in config["category_map"].keys():
        # TODO call trained model instead of training here
        predictor = OscarPredictor(
            model_category=model_category,
            model_type=args.model_type,
            new_season=args.year,
            config=config,
        )
        predictor.train_model()

        # Call model
        predictor.predict_new_season()

        # Save results
        category_res = predictor.df_res_new
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
