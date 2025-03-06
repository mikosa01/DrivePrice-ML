import os 

class Config: 
    SECRET_KEY = os.getenv('SECRET-KEY', 'my_secret_key')
    DEBUG = False
    TESTING = False
    JSONIFY_PRETTYPRINT_REGULAR = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    DEBUG = True
    TESTING = True

class ProductionConfig(Config):
    DEBUG = False 


config_dict = {
    'development' : DevelopmentConfig, 
    'testing' : TestingConfig, 
    'production' : ProductionConfig
}