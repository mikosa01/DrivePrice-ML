from pathlib import Path
from classification import class_model


# directory
PACKAGE_ROOT = Path(class_model.__file__).parent
DATA_DIRECTORY = PACKAGE_ROOT/'data'
TRAINED_MODEL_DIRECTORY = PACKAGE_ROOT/'trained_model'


# data 
DATA_NAME = 'car_price_dataset.csv'
RANDOM_SIZE = 30
TEST_SIZE = 0.25


# features 
COLUMNS = [
            'Brand', 
            'Model', 
            'Year', 
            'Engine_Size', 
            'Fuel_Type', 
            'Transmission',
            'Mileage', 
            'Doors', 
            'Owner_Count', 
            'Price'
        ]

NUMERICAL_COLUMNS = [
                        'Year', 
                        'Engine_Size', 
                        'Mileage'
                    ]

DISCRETE_COLUMNS = [
                        'Doors', 
                        'Owner_Count'
                    ]


OBJECT_COLUMNS = [
                        'Brand', 
                        'Model', 
                        'Fuel_Type', 
                        'Transmission'
                ]

YEAR_COLUMN = 'Year'

CURRENT_YEAR_COLUMNN = 'current_year'

AGE_COLUMN = 'Age'

LABEL_ENCODER_COLUMNS = [
                            'Transmission',
                            'Brand', 
                            'Model', 
                            'Fuel_Type'
                       ]

FEATS_COLUMNS = [
            'Brand', 
            'Model', 
            'Year', 
            'Engine_Size', 
            'Fuel_Type', 
            'Transmission',
            'Mileage', 
            'Doors', 
            'Owner_Count'
        ]

FEATURE_COLUMNS = [ 'Year', 
                   'Engine_Size', 
                   'Transmission', 
                   'Mileage', 
                   'Doors',
                   'Owner_Count', 
                   'Brand_BMW', 
                   'Brand_Chevrolet', 
                   'Brand_Ford',
                   'Brand_Honda', 
                   'Brand_Hyundai', 
                   'Brand_Kia', 
                   'Brand_Mercedes',
                   'Brand_Toyota', 
                   'Brand_Volkswagen', 
                   'Model_5 Series',
                   'Model_A3',
                   'Model_A4', 
                   'Model_Accord', 
                   'Model_C-Class', 
                   'Model_CR-V',
                   'Model_Camry', 
                   'Model_Civic', 
                   'Model_Corolla', 
                   'Model_E-Class',
                   'Model_Elantra', 
                   'Model_Equinox', 
                   'Model_Explorer', 
                   'Model_Fiesta',
                   'Model_Focus', 
                   'Model_GLA', 
                   'Model_Golf', 
                   'Model_Impala',
                   'Model_Malibu', 
                   'Model_Optima', 
                   'Model_Passat', 
                   'Model_Q5',
                   'Model_RAV4', 
                   'Model_Rio', 
                   'Model_Sonata', 
                   'Model_Sportage',
                   'Model_Tiguan', 
                   'Model_Tucson', 
                   'Model_X5', 
                   'Fuel_Type_Electric',
                   'Fuel_Type_Hybrid', 
                   'Fuel_Type_Petrol', 
                   'Age']

TARGET_COLUMNS = 'Price'

# model
MODEL_NAME = 'linear_regression'
PIPELINE_NAME = f'{MODEL_NAME}_output'