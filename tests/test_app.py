from bridge_detector_api.app import create_rest_server
from bridge_detector_api.settings import DevConfig
from bridge_detector_api.api.rest_api import blueprint


def test_create_app_returns_app_instance():
    app = create_rest_server()
    assert app is not None
    assert app.name == 'bridge_detector_api.app'  


def test_create_app_uses_correct_config():
    app = create_rest_server(config_object=DevConfig)
    assert app.config['DEBUG'] is True  


def test_create_app_registers_blueprint():
    app = create_rest_server()
    assert blueprint.name in app.blueprints 

