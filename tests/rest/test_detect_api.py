from PIL import Image
from io import BytesIO
import zipfile
import os


## /detect

def test_detect_endpoint_valid_image(client):
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    data = {
        'image': (img_byte_arr, 'test.png')
    }

    response = client.post('/detect', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    assert response.content_type == 'image/png'


def test_detect_endpoint_no_file(client):
    response = client.post('/detect')
    assert response.status_code == 400
    assert response.json == {"error": "No image uploaded"}


def test_detect_endpoint_empty_filename(client):
    data = {
        'image': (BytesIO(), '')
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {"error": "File is invalid"}


def test_detect_endpoint_invalid_file(client):
    data = {
        'image': (BytesIO(b"not_an_image"), 'test.txt')
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 500
    assert "error" in response.json
    
    
    
## /detect_batch_save

def test_detect_batch_save_valid_zip(client):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        img = Image.new('RGB', (10, 10), color='blue')
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        zip_file.writestr('test_image.png', img_byte_arr.getvalue())
    zip_buffer.seek(0)

    data = {
        'images': (zip_buffer, 'test.zip')
    }

    response = client.post('/detect-batch-save', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    assert "path" in response.json
    assert os.path.exists(response.json['path'])


def test_detect_batch_save_invalid_zip(client):
    data = {
        'images': (BytesIO(b"not_a_zip"), 'test.zip')
    }

    response = client.post('/detect-batch-save', data=data, content_type='multipart/form-data')

    assert response.status_code == 400
    assert response.json == {"error": "Invalid ZIP file"}


def test_detect_batch_save_no_zip_file(client):
    response = client.post('/detect-batch-save')
    assert response.status_code == 400
    assert response.json == {"error": "No file uploaded"}


def test_detect_batch_save_invalid_extension(client):
    data = {
        'images': (BytesIO(b"fake_zip"), 'test.txt')
    }

    response = client.post('/detect-batch-save', data=data, content_type='multipart/form-data')

    assert response.status_code == 400
    assert response.json == {"error": "Only ZIP files are allowed"}
    