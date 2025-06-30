import pytest


from bridges_detection_api.app import create_app
from bridges_detection_api.settings import TestConfig


@pytest.yield_fixture(scope='function')
def app():
    return create_app(TestConfig)