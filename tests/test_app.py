from bridges_detection_api.app import create_app
from bridges_detection_api.settings import DevConfig
from bridges_detection_api.rest.detect_api import blueprint


def test_create_app_returns_app_instance():
    """Проверяет, что create_app возвращает экземпляр Flask"""
    app = create_app()
    assert app is not None
    assert app.name == 'bridges_detection_api.app'  


def test_create_app_uses_correct_config():
    """Проверяет, что конфиг загружается корректно"""
    app = create_app(config_object=DevConfig)
    assert app.config['DEBUG'] is True  


def test_create_app_registers_blueprint():
    """Проверяет, что блюпринт зарегистрирован"""
    app = create_app()
    assert blueprint.name in app.blueprints 

