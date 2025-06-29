from flask import Flask
from bridges_detection_api.rest.detect import api_bp  
from bridges_detection_api.settings import DevConfig 


def create_app(config_object=DevConfig):
    app = Flask(__name__)
    app.config.from_object(config_object)

    app.register_blueprint(api_bp, url_prefix='/api')

    return app