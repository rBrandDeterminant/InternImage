from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename
from flask import Response, send_file
import torch
import json
import base64
import numpy as np
import cv2

import get_bboxes as gb

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'json'}

CONFIG = "configs/coco/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.py"
CKPT = "checkpoint_dir/det/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.pth"

SCORE_TRESHOLD = 0.3
ASYNC_TEST = 'store_true'

PALETTES = ['coco', 'voc', 'citys', 'random']

OUT = "demo"
JSON_DIR = "Json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

class MODEL():
    def __init__(self, config, CKPT, device=DEVICE):
        self.model = gb.init_detector(config, CKPT, device)
        self.b_boxes = {}

    def processing(self, img_path):
        self.b_boxes = gb.get_bboxes(self.model, img_path, SCORE_TRESHOLD)

    def send_bboxes(self, name):
        json_data = json.dumps(self.b_boxes)

        return Response(json_data, 
                        mimetype='application/json',
                        headers={'Content-Disposition':f'attachment;filename={name}'}
        )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/uploads/Json/<name>')
def download_file(path):
    return send_file(path, as_attachment=True)

@app.route('/', methods=['POST', 'GET'])
def process_API():
    '''
    Input format :
        - LOCAL : Image with ALLOWED_EXTENSIONS

        - API : Json {
            "image_name" : Name of the image with ALLOWED_EXTENSIONS -> str
            "image_base64" : string of the image converted to base64 -> str
        }
    '''
    if request.method == 'POST':
        NAME = ""

        # Local Inférence
        if 'file' in request.files:
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if  file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)

                file_bytes = np.fromstring(file.read(), np.uint8)
                img_array = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

                NAME = filename.split(".")[-2] + ".json"

                model.processing(img_path=img_array)
                #result = model.send_bboxes(NAME)

        # API inférence
        else:
            if 'image_base64' not in request.json:
                flash('The file is not an image nor a json')
                return redirect(request.url)
            file = request.json['image_base64']

            if  request.json['image_name'] == '':
                flash('No selected file')
                return redirect(request.url)
            
            if file and allowed_file(request.json['image_name']):
                filename = secure_filename(request.json['image_name'])

                image_encoded = bytes(file, "utf-8")
                base64_decoded = base64.b64decode(image_encoded)
                img_array = np.frombuffer(base64_decoded, dtype=np.uint8)
                img_array = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)

                NAME = filename.split(".")[-2] + ".json"

                model.processing(img_path=img_array)
                #result = model.send_bboxes(NAME)
            
        return model.send_bboxes(NAME)

    return '''
    <!doctype html>
    <title>Upload Image to Process</title>
    <h1>Upload Image to Process</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Process>
    </form>
    '''

if __name__ == '__main__':
    model = MODEL(CONFIG, CKPT, device=DEVICE)

    app.run(debug=True)