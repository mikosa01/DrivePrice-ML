import mlflow
import mlflow.sklearn
from dotenv import load_dotenv, find_dotenv
import os 
from config import config


def model_tracking(*, model):
    load_dotenv()
    model_url = os.environ.get('azure_uri_model')
    mlflow.set_tracking_uri(model_url)
    mlflow.set_experiment("/my-experiment")
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, 'model')