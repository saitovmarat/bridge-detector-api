import os
import zipfile
from flask import Blueprint, request, send_file
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('./best_weights.pt')  
OUTPUT_DIR = './output/results'

api_bp = Blueprint('api', __name__)
     

@api_bp.route('/detect', methods=['POST'])
def detect_bridge():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = model(img_bgr)
    annotated_img = img_bgr.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = box.cls[0]
            if conf > 0.2:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(annotated_img_rgb)

    img_io = io.BytesIO()
    img_pil.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@api_bp.route('/detect-batch-save', methods=['POST'])
def detect_batch_save():
    if 'images' not in request.files:
        return "No file uploaded", 400

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file = request.files['images']
    if file.filename is None:
        return "No filename provided", 400

    if file.filename.endswith('.zip'):
        with io.BytesIO(file.read()) as zip_data:
            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                for filename in zip_ref.namelist():
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with zip_ref.open(filename) as img_file:
                            img_bytes = io.BytesIO(img_file.read())
                            img = Image.open(img_bytes).convert('RGB')
                            img_np = np.array(img)
                            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                            results = model(img_bgr)
                            annotated_img = img_bgr.copy()

                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    conf = box.conf[0]
                                    cls = box.cls[0]
                                    if conf > 0.2:
                                        label = f'{model.names[int(cls)]} {conf:.2f}'
                                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(annotated_img, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(annotated_img_rgb)
                            img_byte_arr = io.BytesIO()
                            pil_img.save(img_byte_arr, format='JPEG')

                            new_name = os.path.splitext(filename)[0] + '_detected.jpg'
                            output_path = os.path.join(OUTPUT_DIR, new_name)
                            with open(output_path, 'wb') as f_out:
                                f_out.write(img_byte_arr.getvalue())

                return {"message": "Images saved successfully", "path": os.path.abspath(OUTPUT_DIR)}, 200

    else:
        return "Only ZIP files are allowed", 400