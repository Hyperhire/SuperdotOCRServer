from model_utils.init import init
from tools.program import load_config, check_gpu
from tools.infer_e2e import extract_all_text

from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process

from flask_restx import Api, Resource, reqparse
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify, abort
from werkzeug.utils import secure_filename

import os
import numpy as np
import paddle
import json

app = Flask(__name__)
CORS(app)
api = Api(app)

config = load_config('configs/e2e/pet_receipts.yml')
use_gpu = config['Global']['use_gpu']
global_config = config['Global']
check_gpu(use_gpu)

model, config = init()
model.eval()

post_process_class = build_post_process(config['PostProcess'],
                                        global_config)
transforms = []
for op in config['Eval']['dataset']['transforms']:
    op_name = list(op)[0]
    if 'Label' in op_name:
        continue
    elif op_name == 'KeepKeys':
        op[op_name]['keep_keys'] = ['image', 'shape']
    transforms.append(op)

ops = create_operators(transforms, global_config)

@api.documentation
def custom_ui():
    return render_template(
        "swagger-ui.html", title=api.title, specs_url='/static/swagger.json'
    )


@api.route("/OCR_img_upload")
class OCR(Resource):
    def post(file):
        f = request.files['upload']
        if not os.path.isdir('./upload_data'):
            os.mkdir('./upload_data')

        f.save(os.path.join('./upload_data', secure_filename(f.filename)))
        f_path = (os.path.join('./upload_data', secure_filename(f.filename)))
        with open(f_path, 'rb') as img:
            image = img.read()
            data = {'image': image}
        batch = transform(data, ops)
        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = paddle.to_tensor(images)
        preds = model(images)

        post_result = post_process_class(preds, shape_list)
        dt_boxes, strs = post_result['points'], post_result['texts']
        output_json, _, __, ___ = extract_all_text(dt_boxes, strs, global_config['targets'])

        return {'result': output_json}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
