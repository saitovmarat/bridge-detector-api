from flask import Flask
from bridges_detection_api.rest import detect
from typing import Type
from bridges_detection_api.settings import Config


def create_app(config_object: Type[Config]):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.register_blueprint(detect.blueprint, url_prefix='/api')
    return app