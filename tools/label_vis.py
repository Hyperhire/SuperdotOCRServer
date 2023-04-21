
import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import re
from PIL import ImageFont, ImageDraw, Image


def draw_label(src_img, label_point_str, img_path, output_path):
    if len(label_point_str) > 0:
        temp = Image.fromarray(src_img)
        draw = ImageDraw.Draw(temp)

        strs = []
        boxes = []
        for item in label_point_str:
            str = item['transcription']
            box = item['points']
            strs.append(str)
            boxes.append(box)

        for str, box in zip(np.array(strs), np.array(boxes)):
            box = box.astype(np.int32).reshape((-1, 1, 2))
            ################################################################
            draw.text(box[3][0], str, font=ImageFont.truetype('NanumBarunGothic.ttf', 15), fill=(0, 50, 100))
            draw.polygon([tuple(box[0][0]), tuple(box[1][0]), tuple(box[2][0]), tuple(box[3][0])], outline=(0, 0, 255))
            ################################################################

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = os.path.join(output_path, os.path.basename(img_path))
        result = np.array(temp)
        cv2.imwrite(save_path, result)


def main(input_path, output_path):

    with open(input_path + '/Label.txt', 'r') as f:
        labels = f.readlines()

    for label in labels:
        img_path = re.sub('train', '', label.split('\t')[0])
        print(img_path)
        label_point_str = json.loads(label.split('\t')[1])
        src_img = cv2.imread(input_path + img_path)
        draw_label(src_img, label_point_str, img_path, output_path)

input_path = './pet_receipts/train'
output_path = './pet_receipts/train_vis'

main(input_path, output_path)
