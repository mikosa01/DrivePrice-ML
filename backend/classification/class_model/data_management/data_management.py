import pandas as pd 
import numpy as np 
import joblib
import errors
from pathlib import Path
from class_model.config import config


def load_dataset(*, file_name:str) -> pd.DataFrame:
    file_path = Path(config.DATA_DIRECTORY)
    data = pd.read_csv(file_path/file_name)
    return data 

def save_pipeline(*, pipeline_persit:object) -> None:
    pipeline_name  = f'{config.PIPELINE_NAME}.pkl'
    pipe_dir = Path(config.TRAINED_MODEL_DIRECTORY)
    pipeline_directory = pipe_dir/pipeline_name
    joblib.dump(pipeline_persit, pipeline_directory)


def load_pipeline(*, load_model:str) -> object:
    
    model_path = Path(config.TRAINED_MODEL_DIRECTORY)/f'{load_model}.pkl'
    if not model_path.exists():
        raise errors.CustomFileNotFoundError(f'{model_path} not found in the directory')
    return joblib.load(filename=model_path)