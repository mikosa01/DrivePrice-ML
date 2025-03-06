from flask import Flask
import os 
from .routes import main
from .config import config_dict
from flask_cors import CORS

def create_app(config_name='development'):
    app = Flask(__name__)
    app.config.from_object(config_dict[config_name])
    CORS(app) 
    app.register_blueprint(main)

    return app



if __name__ == '__main__':
    port =int(os.environ.get('PORT', 5000))
    create_app(config_name='development').run(host='0.0.0.0', port=port)