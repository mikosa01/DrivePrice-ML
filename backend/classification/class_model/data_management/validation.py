from class_model.config import config
import pandas as pd 

def validation_input(input_data:pd.DataFrame) -> pd.DataFrame:
    input_data = input_data.copy()

    if input_data[config.NUMERICAL_COLUMNS].isnull().any().any():
        input_data.dropna(subset= config.NUMERICAL_COLUMNS, inplace=True)

    if input_data[config.OBJECT_COLUMNS].isnull().any().any():
        input_data.dropna(subset=config.OBJECT_COLUMNS, inplace=True)

    if input_data[config.DISCRETE_COLUMNS].isnull().any().any():
        input_data.dropna(subset=config.DISCRETE_COLUMNS, inplace=True)
    
    return input_data

    