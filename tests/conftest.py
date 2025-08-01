import pytest


from bridges_detection_api.app import create_rest_server
from bridges_detection_api.settings import TestConfig


@pytest.fixture(scope='function')
def app():
    return create_rest_server(TestConfig)