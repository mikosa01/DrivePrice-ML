import pandas as pd 
import numpy as np 
import joblib
from classification.class_model.data_management import errors
from pathlib import Path
from classification.class_model.config import config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    Parameters
    ----------
    file_name : str
        The name of the CSV file to load.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.
    """
    # Set the directory path where the dataset is stored.
    file_path = Path(config.DATA_DIRECTORY)
    # Read the CSV file and return it as a pandas DataFrame.
    data = pd.read_csv(file_path / file_name)
    return data

def save_pipeline(*, pipeline_persit: object) -> None:
    """
    Save a trained pipeline object to a file.

    Parameters
    ----------
    pipeline_persit : object
        The trained pipeline object to save.

    Returns
    -------
    None
    """
    # Set the pipeline file name using the configured pipeline name.
    pipeline_name = f'{config.PIPELINE_NAME}.pkl'
    # Set the directory path where the trained pipeline will be stored.
    pipe_dir = Path(config.TRAINED_MODEL_DIRECTORY)
    # Create the full path for saving the pipeline.
    pipeline_directory = pipe_dir / pipeline_name
    # Save the pipeline object to the specified file path.
    joblib.dump(pipeline_persit, pipeline_directory)

def load_pipeline(*, load_model: str) -> object:
    """
    Load a previously saved model pipeline.

    Parameters
    ----------
    load_model : str
        The name of the model pipeline to load.

    Returns
    -------
    object
        The loaded model pipeline object.

    Raises
    ------
    errors.CustomFileNotFoundError
        If the specified model file is not found.
    """
    # Construct the full file path of the model to be loaded.
    model_path = Path(config.TRAINED_MODEL_DIRECTORY) / f'{load_model}.pkl'
    # Check if the model file exists at the specified path, if not raise an error.
    if not model_path.exists():
        raise errors.CustomFileNotFoundError(f'{model_path} not found in the directory')
    # Load and return the model object from the file.
    return joblib.load(filename=model_path)
