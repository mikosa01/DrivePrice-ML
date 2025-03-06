import pytest
import json
import pandas as pd 
from classification.class_model.data_management.data_management import load_dataset 
from classification.class_model.predict import make_prediction





def test_websites_endpoint_200(flask_test_client):

    response = flask_test_client.get('/home')

    assert response.status_code == 200


def test_predict_endpoint(flask_test_client):
    input_data = {
    'Brand': 'Volkswagen',
    'Model': 'Golf', 
    'Year': 2001,
    'Engine_Size': 2.1,
    'Fuel_Type': 'Petrol',
    'Transmission': 'Automatic', 
    'Mileage': 157882,
    'Doors': 4,
    'Owner_Count': 3
}
    
    response = flask_test_client.post('/predict', data=input_data)

    assert response.status_code == 200
    assert b'4334.028' in response.data
