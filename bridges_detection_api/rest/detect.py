import os
from PIL import Image
import zipfile
import cv2
from flask import Blueprint, jsonify, request, send_file
import io
from ultralytics import YOLO
from bridges_detection_api.use_cases.annotate import annotated_image
from bridges_detection_api.use_cases.preprocess_img import preprocess_image
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
        img_bgr = preprocess_image(file.stream)
        detections = detected_objects(img_bgr, model)
        annotated_img = annotated_image(img_bgr, detections)
        
        annotated_pil = Image.fromarray(annotated_img)
        img_io = io.BytesIO()
        annotated_pil.save(img_io, format='JPEG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

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
            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                for filename in zip_ref.namelist():
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with zip_ref.open(filename) as img_file:
                            img_bgr = preprocess_image(img_file)
                            detections = detected_objects(img_bgr, model)
                            annotated_img = annotated_image(img_bgr, detections)

                            new_name = os.path.splitext(filename)[0] + '_detected.jpg'
                            output_path = os.path.join(OUTPUT_DIR, new_name)

                            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                            annotated_pil.save(output_path, format='JPEG')

        return jsonify({
            "message": "Images saved successfully",
            "path": os.path.abspath(OUTPUT_DIR)
        }), 200

    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid ZIP file"}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500