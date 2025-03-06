import pytest

from class_api.api.config import config_dict
from class_api.api.app import create_app

@pytest.fixture
def app():
    app = create_app(config_name='testing')
    with app.app_context():
        yield app

@pytest.fixture
def flask_test_client(app):
    with app.test_client() as test_client:
        yield test_client