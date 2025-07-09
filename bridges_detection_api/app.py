from flask import Flask
from bridges_detection_api.rest import detect_api
from typing import Type
from bridges_detection_api.settings import Config
from bridges_detection_api.settings import DevConfig


def create_app(config_object: Type[Config]=DevConfig):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.register_blueprint(detect_api.blueprint, url_prefix='/api')
    return app