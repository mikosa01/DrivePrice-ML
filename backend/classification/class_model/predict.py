from classification.class_model.config import config
from classification.class_model.data_management.data_management import load_pipeline
from classification.class_model.data_management import errors
from classification.class_model.data_management.validation import validation_input
from classification.class_model.pipeline import pipeline
import pandas as pd 
import numpy as np
import typing as t 


my_pipe = load_pipeline(load_model=config.PIPELINE_NAME)

def make_prediction(*, input_data:t.Union[pd.DataFrame, dict]) -> dict:  
    if isinstance (input_data, dict):
        data = pd.DataFrame([input_data])
    else:
        data = input_data.copy()
    
    required_columns = set(config.FEATS_COLUMNS)  # Features expected in input
    missing_columns = required_columns - set(data.columns)  # Find missing columns

    if missing_columns:
        raise errors.CustomValueError(f"Missing columns in input data: {', '.join(missing_columns)}")

    
    validated_data = validation_input(data)
    y_pred = my_pipe.predict(validated_data)

    return {'prediction': y_pred.tolist()}