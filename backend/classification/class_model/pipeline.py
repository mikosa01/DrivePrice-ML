from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from classification.class_model.data_management import preprocess as pre
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from classification.class_model.config import config



pipeline = Pipeline(
    [
        ("label_encoder", pre.LabelEncoder(variables=config.LABEL_ENCODER_COLUMNS)), 
        # ("onehot_encoder", pre.OneHotEncoding(variables=config.ONE_HOT_ENCODER_COLUMNS)),
        ("age_columns", pre.AgeCalculator(varaible=config.YEAR_COLUMN, colName=config.AGE_COLUMN)), 
        ('standard_scaler', StandardScaler()),
        ('model', LinearRegression()) 
    ]
)