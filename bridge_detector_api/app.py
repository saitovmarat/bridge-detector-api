from typing import Type
from flask import Flask
from .api import rest_api
from .api.udp_api import run_udp_server
from .settings import Config, DevConfig


def create_rest_server(config_object: Type[Config]=DevConfig):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.register_blueprint(rest_api.blueprint)
    return app


def create_udp_server(host: str="0.0.0.0", port: int=9999):
    run_udp_server(host, port)
