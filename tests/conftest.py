import pytest


from bridge_detector_api.app import create_rest_server
from bridge_detector_api.settings import TestConfig


@pytest.fixture(scope='function')
def app():
    return create_rest_server(TestConfig)