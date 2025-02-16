import math
import pandas as pd
import numpy as np
import time
from classification.class_model.pipeline import pipeline
from datetime import datetime
from classification.class_model.data_management.preprocess import AgeCalculator
from classification.class_model.config import config
from classification.class_model.data_management.data_management import load_dataset
from classification.class_model.predict import make_prediction

def test_single_prediction():
    X_train = pd.DataFrame(
)
    input_data = pd.DataFrame({
    'Brand': ['Volkswagen', 'Volkswagen', 'Volkswagen', 'Volkswagen'],
    'Model': ['Tiguan', 'Tiguan', 'Passat', 'Golf'],
    'Year': [2001, 2005, 2010, 2015],
    'Engine_Size': [2.1, 2.0, 2.2, 1.8],
    'Fuel_Type': ['Diesel', 'Diesel', 'Petrol', 'Diesel'],
    'Transmission': ['Manual', 'Automatic', 'Manual', 'Semi-Automatic'],
    'Mileage': [157882, 120000, 98000, 75000],
    'Doors': [3, 5, 4, 3],
    'Owner_Count': [3, 2, 1, 2]
})

    # test_data = input_data.drop(config.TARGET_COLUMNS, axis =1)
    my_pred = make_prediction(input_data=input_data)

    assert my_pred is not None
    assert isinstance(my_pred.get('prediction')[0], float)
    # assert math.ceil(my_pred.get('prediction')[0]) == 8457 


def test_age_calculator(): 
    transformer = AgeCalculator(varaible=["year"], colName="age")
    test_data = pd.DataFrame({"year": [2000, 2010, 2020]})
    current_year = datetime.now().year
    
    transformed_data = transformer.transform(test_data)
    
    assert "age" in transformed_data.columns
    assert transformed_data["age"].tolist() == [current_year - 2000, current_year - 2010, current_year - 2020]

def test_pipeline_fit_predict():

    X_train = pd.DataFrame({
    'Brand': ['Volkswagen', 'Volkswagen', 'Volkswagen', 'Volkswagen'],
    'Model': ['Tiguan', 'Tiguan', 'Passat', 'Golf'],
    'Year': [2001, 2005, 2010, 2015],
    'Engine_Size': [2.1, 2.0, 2.2, 1.8],
    'Fuel_Type': ['Diesel', 'Diesel', 'Petrol', 'Diesel'],
    'Transmission': ['Manual', 'Automatic', 'Manual', 'Automatic'],
    'Mileage': [157882, 120000, 98000, 75000],
    'Doors': [3, 5, 4, 3],
    'Owner_Count': [3, 2, 1, 2]
}
)
    y_train = pd.Series([3342, 4500, 6800, 9000])
    pipeline.fit(X_train, y_train)   
    y_pred = pipeline.predict(X_train)
    assert len(y_pred) == len(y_train)

def test_pipeline_speed():
    X_train = pd.DataFrame({
    'Brand': ['Volkswagen', 'Volkswagen', 'Volkswagen', 'Volkswagen'],
    'Model': ['Tiguan', 'Tiguan', 'Passat', 'Golf'],
    'Year': [2001, 2005, 2010, 2015],
    'Engine_Size': [2.1, 2.0, 2.2, 1.8],
    'Fuel_Type': ['Diesel', 'Diesel', 'Petrol', 'Diesel'],
    'Transmission': ['Manual', 'Automatic', 'Manual', 'Automatic'],
    'Mileage': [157882, 120000, 98000, 75000],
    'Doors': [3, 5, 4, 3],
    'Owner_Count': [3, 2, 1, 2]
}
)
    y_train = pd.Series([3342, 4500, 6800, 9000])
    start_time = time.time()
    make_prediction(input_data=X_train)
    end_time = time.time()
    
    assert end_time - start_time < 5  # Ensure training completes in under 5 seconds



