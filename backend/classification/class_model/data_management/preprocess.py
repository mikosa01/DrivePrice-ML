from classification.class_model.data_management import errors
import typing as t 
import pandas as pd
import numpy as np 
from datetime import datetime
from sklearn.base import TransformerMixin, BaseEstimator

class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        """Initialize with variables to be label-encoder encoded."""
        if not isinstance(variables, list):
            self.variables = [variables]
        else: 
            self.variables = variables
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'LabelEncoder':
        self.encoder= {}
        temp = pd.concat([X,y], axis = 1)
        temp.columns= list(X.columns) + ["target"]
        for col in self.variables:
            label_encoder = temp.groupby([col])['target'].mean().sort_values(ascending=True).index
            self.encoder[col] = {k:i for i, k in enumerate(label_encoder, 0)}

        return self 
    
    def transform(self, X:pd.DataFrame):
        """Transform the data by applying label-encoder encoding."""
        X = X.copy() # Avoid modifying the original DataFrame
        for feature in self.variables:
            if feature not in X.columns:
                raise errors.CustomValueError(f"Feature '{feature}' not found in the input data.")
            X[feature] = X[feature].map(self.encoder[feature])

                # check if transformer introduces NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {
                key: value for (key, value) in null_counts.items() if value is True
            }
            raise errors.CustomInvalidModelInput(
                f"Categorical encoder has introduced NaN when "
                f"transforming categorical variables: {vars_.keys()}"
            )
        return X
    
    def fit_transform(self, X:pd.DataFrame, y:pd.Series= None) -> pd.DataFrame:
        """Fit and transform the data (combine fit and transform)."""
        self.fit(X, y)
        return self.transform(X)
    

class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None): 
        """Initialize with variables to be one-hot encoded."""
        if isinstance(variables, list):
            self.variables = variables
        else: 
            self.variables = [variables]
    
    def fit (self, X:pd.DataFrame, y:pd.Series) -> 'OneHotEncoding': 
        """Fit method (no-op for OneHotEncoding)"""
        return self
    
    def transform(self, X:pd.DataFrame):
        """Transform the data by applying one-hot encoding."""
        X = X.copy() # Avoid modifying the original DataFrame
        for feature in self.variables:
            if feature not in X.columns: 
                raise errors.CustomValueError(f"Feature '{feature}' not found in the input data.")
        X = pd.get_dummies(X, columns=self.variables, drop_first=True).astype('int')
        return X
    
    def fit_transform(self, X:pd.DataFrame, y:pd.Series = None):
        """Fit and transform the data (combine fit and transform)."""
        self.fit(X, y)
        return self.transform(X)
    

class AgeCalculator(BaseEstimator, TransformerMixin): 
    def __init__(self, varaible = None, colName=None ) :
        """
        Transformer to calculate age from a given birth year.

        Parameters:
        - variable (str or list): Column(s) containing birth year.
        - colName (str): Name of the new column to store calculated age.
        """
        self.current_year = datetime.now().year
        self.colName = colName
        if isinstance (varaible, list):
            self.varaible = varaible
        else:
            self.varaible = [varaible]

    def fit (self, X:pd.DataFrame, y:pd.Series=None):

        return self
    
    def transform(self, X:pd.DataFrame):
        """Transforms the dataset by calculating age."""
        X = X.copy() # Avoid modifying the original DataFrame
        for feature in self.varaible:
            if feature not in X.columns: 
                raise errors.CustomValueError(f"Feature '{feature}' not found in the input data.")
        X[self.colName] =  self.current_year - X[feature]
        return X
    
    def fit_transform(self, X, y = None, **fit_params):
        """Combines fit and transform."""
        self.fit(X, y)
        return self.transform(X)