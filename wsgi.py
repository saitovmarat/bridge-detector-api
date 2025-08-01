# from bridges_detection_api.app import create_app
from bridges_detection_api.app import create_udp_server

# app = create_app()

if __name__ == "__main__":
#   app.run(debug=True, host='0.0.0.0', port=5000)
    create_udp_server(host="0.0.0.0", port=9999)