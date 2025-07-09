from bridges_detection_api.app import create_app
from bridges_detection_api.settings import DevConfig

app = create_app(config_object=DevConfig)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)