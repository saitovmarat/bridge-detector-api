import base64
import os
from PIL import Image
from zipfile import ZipFile, BadZipFile
from flask import Blueprint, jsonify, request, send_file
import io
from ultralytics import YOLO
from bridges_detection_api.use_cases.annotate import annotated_image
from bridges_detection_api.use_cases.img_preprocessor import preprocess_image_dto
from bridges_detection_api.use_cases.detect import detected_objects


model = YOLO('./best_weights.pt')  
blueprint = Blueprint('api', __name__)


@blueprint.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    if not file:
        return jsonify({"error": "File is invalid"}), 400

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_dto = preprocess_image_dto(file.stream)
        detections = detected_objects(img_dto, model)
        annotated_img_dto = annotated_image(img_dto, detections)
        
        image_bytes = base64.b64decode(annotated_img_dto.image_data)
        img_io = io.BytesIO(image_bytes)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@blueprint.route('/detect-batch-save', methods=['POST'])
def detect_batch_save():
    if 'images' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['images']
    if not file or file.filename == '':
        return jsonify({"error": "Invalid or empty filename"}), 400

    if file.filename is None:
        return jsonify({"error": "Filename is missing"}), 400

    if not file.filename.endswith('.zip'):
        return jsonify({"error": "Only ZIP files are allowed"}), 400

    OUTPUT_DIR = './output/results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        with io.BytesIO(file.read()) as zip_data:
            with ZipFile(zip_data, 'r') as zip_ref:
                annotated_images = []
                for filename in zip_ref.namelist():
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with zip_ref.open(filename) as img_file:
                            img_dto = preprocess_image_dto(img_file)
                            detections = detected_objects(img_dto, model)
                            annotated_img_dto = annotated_image(img_dto, detections)
                            
                            annotated_images.append(annotated_img_dto)
            
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for idx, annotated_img_dto in enumerate(annotated_images):
            image_bytes = base64.b64decode(annotated_img_dto.image_data)
            img_io = io.BytesIO(image_bytes)
            annotated_pil = Image.open(img_io)

            output_path = os.path.join(OUTPUT_DIR, f"image_{idx}_annotated.jpg")
            annotated_pil.save(output_path, format='JPEG')
            
        return jsonify({
            "message": "Images saved successfully",
            "path": os.path.abspath(OUTPUT_DIR)
        }), 200

    except BadZipFile:
        return jsonify({"error": "Invalid ZIP file"}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500