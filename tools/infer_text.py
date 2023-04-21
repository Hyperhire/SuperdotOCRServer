# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import init_model
from ppocr.utils.utility import get_image_file_list
import tools.program_inference as program

from PIL import ImageFont, ImageDraw, Image

def rectify(box):

    # y-axis error
    if box[0, 1] > box[3, 1] and box[1, 1] < box[2, 1]:
        box[0, 1] = box[1, 1]
        box[3, 1] = box[2, 1]
    if box[0, 1] < box[3, 1] and box[1, 1] > box[2, 1]:
        box[1, 1] = box[0, 1]
        box[2, 1] = box[3, 1]

    # x-axis error
    if box[0, 0] > box[1, 0] and box[2, 1] > box[3, 1]:
        box[0, 0] = box[3, 0]
        box[1, 0] = box[2, 0]
    if box[0, 0] < box[1, 0] and box[2, 1] < box[3, 1]:
        box[2, 0] = box[1, 0]
        box[3, 0] = box[0, 0]

    upper_y = np.mean([box[1, 1], box[0, 1]])
    lower_y = np.mean([box[3, 1], box[2, 1]])
    left_x = np.mean([box[0, 0], box[3, 0]])
    right_x = np.mean([box[1, 0], box[2, 0]])
    rect_box = np.array([[left_x, upper_y],
                         [right_x, upper_y],
                         [right_x, lower_y],
                         [left_x, lower_y]])
    return rect_box

def text_e2e_res(dt_boxes, strs, config, img, img_name):
    if len(dt_boxes) > 0:

        # box = dt_boxes[0]
        # string = strs[0]
        # rect_box = rectify(box)

        h, w, _ = img.shape
        tmp_canvas = Image.new('RGB', (2 * w, h), (255, 255, 255))
        temp = Image.fromarray(img)
        tmp_canvas.paste(temp, (w, 0))
        draw = ImageDraw.Draw(tmp_canvas, )
        for box, string in zip(dt_boxes, strs):
            rect_box = rectify(box)
            box = rect_box.astype(np.int32).reshape((-1, 1, 2))
            ################################################################
            # 너무 작은 상자는 없애는게 깔끔하겠는데, 그랬다가는 한음절이 사라질까봐 걱정됨
            ################################################################
            if box.shape[0] < 12:
                draw.text(box[0][0], string, font=ImageFont.truetype('NanumBarunGothic.ttf', 10), fill=(0, 50, 100))
                draw.polygon([tuple(box[0][0]), tuple(box[1][0]), tuple(box[2][0]), tuple(box[3][0])], outline=(0, 0, 200))
                draw.polygon([tuple(box[0][0]+[w,0]), tuple(box[1][0]+[w,0]), tuple(box[2][0]+[w, 0]), tuple(box[3][0]+[w, 0])], outline=(0, 0, 255))
            else:
                draw.text(box[11][0], string, font=ImageFont.truetype('NanumBarunGothic.ttf', 15), fill=(0, 50, 100))
                draw.polygon([tuple(box[0][0]), tuple(box[1][0]), tuple(box[2][0]), tuple(box[3][0]),
                              tuple(box[4][0]), tuple(box[5][0]), tuple(box[6][0]), tuple(box[7][0]),
                              tuple(box[8][0]), tuple(box[9][0]), tuple(box[10][0]), tuple(box[11][0])],
                             outline=(0, 0, 255))
            ################################################################

        save_det_path = os.path.dirname(config['Global']['save_res_path']) + "/e2e_results-Paper_Check_Proj_20220120_ID_test"
        if not os.path.exists(save_det_path):
            os.makedirs(save_det_path)
        save_path = os.path.join(save_det_path, os.path.basename(img_name))
        result = np.array(tmp_canvas)
        cv2.imwrite(save_path, result)
        logger.info("The e2e Image saved in {}".format(save_path))


def main():
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])

    init_model(config, model)

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    # img_name = get_image_file_list(config['Global']['infer_img'])[0]
    with open(save_res_path, "wb") as fout:
        for img_name in get_image_file_list(config['Global']['infer_img']):
            logger.info("infer_img: {}".format(img_name))
            with open(img_name, 'rb') as f:
                image = f.read()
                data = {'image': image}
            batch = transform(data, ops)
            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds, shape_list)
            dt_boxes, strs = post_result['points'], post_result['texts']
            # write resule
            dt_boxes_json = []
            for poly, string in zip(dt_boxes, strs):
                tmp_json = {"transcription": string}
                tmp_json['points'] = poly.tolist()
                dt_boxes_json.append(tmp_json)
            otstr = img_name + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())
            img = cv2.imread(img_name)
            text = text_e2e_res(dt_boxes, strs, config, img, img_name)
    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()

