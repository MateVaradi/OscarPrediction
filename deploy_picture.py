import bentoml
from bentoml import Service
from bentoml.io import PandasDataFrame, JSON
from datetime import datetime
import pandas as pd

from src.model import process_results_df
from src.utils import load_config

config = load_config("configs/config.yml")

# Create BentoML service
class MyModelRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = bentoml.sklearn.load_model(bento_model)

    @bentoml.Runnable.method(batchable=False)
    def predict(self, input_data):
        return self.model.predict_proba(input_data)


# Load the saved model
cat = "Picture"
cat_tag = cat.lower().replace(" ","")
model_type = config["default_model_type"]
model_name = f'{config["model_prefix"]}-{model_type}-{cat_tag}'
bento_model = bentoml.models.get(f"{model_name}:latest")
model_runner = bentoml.Runner(MyModelRunner, models=[bento_model])

# Define a BentoML service
svc = Service("batch_inference_service", runners=[model_runner])

# Define an API for batch prediction
@svc.api(input=JSON(), output=JSON())
def predict_picture(request: dict):
    try:
        # Extract input data and additional argument from the request
        input_data = pd.DataFrame(request["data"])
        predictors = request["vars"]
        X_pred = input_data[predictors]

        # Call predict_proba on the input data
        probs = model_runner.predict.run(X_pred)[:, 1] # predict_proba

        result_df = process_results_df(input_data, probs)

        return result_df
    
    except KeyError as e:
        return {"error": f"Missing key in request payload: {e}"}
    except Exception as e:
        return {"error": str(e)}
