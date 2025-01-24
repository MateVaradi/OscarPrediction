
import bentoml
from datetime import datetime
import pandas as pd


from src.model import OscarPredictor
from src.utils import load_config

config = load_config("configs/config.yml")


def main():
    # Train models
    for cat in config["category_map"].keys():
        for model_type in ["logit", "rf"]:
            predictor = OscarPredictor(model_category=cat, config=config, model_type=model_type)
            predictor.train_model_cv()

            # Save model
            time_tag = datetime.today().strftime("%Y%m%d%H%S")
            cat_tag = cat.lower().replace(" ","")
            model_name = f'{config["model_prefix"]}-{model_type}-{cat_tag}'
            bentoml.sklearn.save_model(model_name, predictor.model)
            print(f"Model saved: {model_name}\n")

    
if __name__ == "__main__":
    main()
