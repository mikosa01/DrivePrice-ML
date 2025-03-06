from flask import Blueprint, render_template, request, url_for, redirect
from class_api.api.validator import CarDetailSchema
from classification.class_model.predict import make_prediction
import pandas as pd

main = Blueprint('main', __name__)

@main.route('/home', methods = ['GET'])
def home (): 
    return render_template('index.html')

@main.route('/')
def redirect_to_home():
    return redirect(url_for('main.home')) 

@main.route('/predict', methods =['POST'])
def predict():
    Brand = request.form['Brand']
    Model= request.form['Model']
    Year = int(request.form['Year'])
    Engine_Size = request.form['Engine_Size']
    Fuel_Type = request.form['Fuel_Type']
    Transmission = request.form['Transmission']
    Mileage =  request.form['Mileage']
    Doors = request.form['Doors']
    Owner_Count = request.form['Owner_Count']

    input_data = {
        'Brand': Brand,
        'Model': Model,
        'Year': Year,
        'Engine_Size': Engine_Size,
        'Fuel_Type': Fuel_Type,
        'Transmission': Transmission,
        'Mileage': Mileage,
        'Doors': Doors,
        'Owner_Count': Owner_Count
    }
   
    # schema = CarDetailSchema()
    # data_val = schema.validate_input(input_data)

    result = make_prediction(input_data=input_data)
    return render_template('results.html', pred=f'The price estimate  for {input_data["Brand"]}, {input_data["Model"]} model is £{result}')
    
    

    

  
