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

# -*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List

import numpy as np
import time

import os
import sys
import copy

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
#from IPython.display import Image
from tqdm import tqdm

from jamo import h2j, j2hcj
from nltk.metrics.distance import edit_distance
from operator import itemgetter
import re


def is_on_same_line(box_a, box_b, smallest_y_overlap = 0.3):

    """Check if two boxes are on the same line by their y-axis coordinates.
    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.
    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line
    Returns:
        The bool flag indicating if they are on the same line
    """

    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # 서로 값 대소 비교
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    # smallest_y_overlap 쓰레쉬 홀드 초과하는지
    if b_y_min <= a_y_max:
        if smallest_y_overlap is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * smallest_y_overlap
            min_b_overlap = (b_y_max - b_y_min) * smallest_y_overlap
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def draw_e2e_res(dt_boxes, strs, config, img, img_name):
    if len(dt_boxes) > 0:

        h, w, _ = img.shape
        tmp_canvas = Image.new('RGB', (2 * w, h), (255, 255, 255))
        temp = Image.fromarray(img)
        tmp_canvas.paste(temp, (w, 0))
        draw = ImageDraw.Draw(tmp_canvas, )

        for box, string in zip(dt_boxes, strs):
            box = box.astype(np.int32).reshape((-1, 1, 2))
            ################################################################
            if box.shape[0] < 12:
                draw.text(box[0][0], string, font=ImageFont.truetype('fonts/NanumBarunGothic.ttf', 10), fill=(0, 50, 100))
                draw.polygon([tuple(box[0][0]), tuple(box[1][0]), tuple(box[2][0]), tuple(box[3][0])], outline=(0, 0, 200))
                draw.polygon([tuple(box[0][0]+[w,0]), tuple(box[1][0]+[w,0]), tuple(box[2][0]+[w, 0]), tuple(box[3][0]+[w, 0])], outline=(0, 0, 255))
            else:
                draw.text(box[11][0], string, font=ImageFont.truetype('fonts/NanumBarunGothic.ttf', 15), fill=(0, 50, 100))
                draw.polygon([tuple(box[0][0]), tuple(box[1][0]), tuple(box[2][0]), tuple(box[3][0]),
                              tuple(box[4][0]), tuple(box[5][0]), tuple(box[6][0]), tuple(box[7][0]),
                              tuple(box[8][0]), tuple(box[9][0]), tuple(box[10][0]), tuple(box[11][0])],
                             outline=(0, 0, 255))

        save_det_path = os.path.join(os.path.dirname(config['Global']['save_res_path']), "image_result")
        if not os.path.exists(save_det_path):
            os.makedirs(save_det_path)
        save_path = os.path.join(save_det_path, os.path.basename(img_name))
        result = np.array(tmp_canvas)
        cv2.imwrite(save_path, result)

        logger.info("The e2e Image saved in {}".format(save_path))


def split_jamo(text):
    return j2hcj(h2j(text))


def find_nearest_line(line_texts, target):

    score_dict = [(line,
                   edit_distance(split_jamo(target), split_jamo(line[:len(target)])) / (len(split_jamo(target)) + 0.000001))
                  for line in line_texts]

    if any(np.array(list(map(itemgetter(1), score_dict))) < 0.3):
        arg = np.argmin(list(map(itemgetter(1), score_dict)))
        return True, arg, score_dict[arg][0]
    else:
        return False, 1000000, 'NaN'


def make_info(res, target):
    try:
        if res[0]:
            tmp_cand = res[2][len(target):]
            if target in ['병원명', '대표자']:
                re_cand = re.compile('[가-힣|()]+').findall(tmp_cand)
                output = ''.join(re_cand)
            elif target in ['사업자 등록번호', '전화번호']:
                re_cand = re.compile('[0-9|\-|\s]+').findall(tmp_cand)
                output = re_cand[0].replace(' ', '')
            elif target in ['날짜', '결제일', '청구일']:
                re_cand = re.compile('[0-9|\-]+').findall(tmp_cand)
                output = re_cand[0]
            elif target in ['총금액', '카드 결제', '부가세', '청구 금액', '1.오늘청구']:
                re_cand = re.compile('[0-9|,]+').findall(tmp_cand)
                output = re_cand[0]
            elif target in ['주소']:
                re_cand = re.compile('[가-힣|0-9|\-|\s]+').findall(tmp_cand)
                output = re_cand[0]
            elif target in ['사업장 소재지']:
                re_cand = re.compile('[가-힣|\s]').findall(tmp_cand)
                output = ''.join(re_cand).strip()
            if tmp_cand != '':
                return output
            else:
                return 'NaN'
        else:
            return 'NaN'
    except:
        return 'NaN'


def information_finder(line_texts, targets, line_bboxes):

    output_json = {'병원명': 'NaN', '사업장 소재지': 'NaN', '전화번호': 'NaN', '담당자': 'NaN',
                   '날짜': 'NaN', '총합계': 'NaN'}

    for target in targets:
        res = find_nearest_line(line_texts, target)
        # print('target', target, '-----', res)
        tmp_output = make_info(res, target)
        line_cnt = 1
        bbox = []

        if target == '사업장 소재지' and tmp_output != 'NaN':
            tmp_output += ' ' + line_texts[res[1]+1]
            line_cnt += 1

        if tmp_output != 'NaN':
            bbox = [line_bboxes[res[1]]]
            if target in ['사업장 소재지', '주소']:
                if line_cnt >= 2:
                    bbox.append(line_bboxes[res[1]+1])
                output_json['사업장 소재지'] = [tmp_output, line_cnt, bbox]

            elif target in ['전화번호']:
                output_json['전화번호'] = [tmp_output, line_cnt, bbox]

            elif target in ['1.오늘청구', '총금액']:
                output_json['총합계'] = [tmp_output, line_cnt, bbox]

            elif target in ['대표자']:
                output_json['담당자'] = [tmp_output, line_cnt, bbox]

            elif target in ['날짜', '청구일', '결제일']:
                date_pattern = re.compile('[0-9]+\-[0-9]+\-[0-9]+')
                if date_pattern.match(tmp_output):
                    output_json['날짜'] = [tmp_output, line_cnt, bbox]

            elif target in ['병원명']:
                output_json['병원명'] = [tmp_output, line_cnt, bbox]

    return output_json


def is_similar(text, targets):
    for target in targets:
        for split_text in text.split(' '):
            score = edit_distance(split_jamo(target), split_jamo(split_text[:len(target)])) / (len(split_jamo(target)) + 0.000001)
            if score < 0.3:
                return True

    return False


def prescript_finder(line_texts: List[str], line_bboxes, receipt_type):
    prescript_start = False
    animals_data = []
    if receipt_type == -1:
        return animals_data

    price_pattern = re.compile('\d+,\d+(,\d+)*')
    date_pattern = re.compile('\d+\-\d+\-\d+')
    is_name_multiline = False #두 줄 이상 이름일 경우 flag를 통해 한 줄로 concat시킨다.

    data = {}
    medicine_data = {"name": '', "num": '', "price": '', "bbox": []}
    medicine_name_temp = ''

    for i, line_text in enumerate(line_texts):
        if receipt_type == 0: #쿨펫 형식 영수증
            if is_similar(line_text, ["동물명"]):

                if prescript_start: #한 영수증에 2마리 이상의 동물이 존재할 때
                    animals_data.append(copy.deepcopy(data))
                    data = {"name": [line_text.replace('[', '').replace(' ', '').replace(']', '').split(':')[-1],
                                     line_bboxes[i]], "medicines": []}
                    continue
                else:
                    prescript_start = True
                    data["name"] = [line_text.replace('[', '').replace(' ', '').replace(']', '').split(':')[-1],
                                    line_bboxes[i]]
                    data["medicines"] = []
                    continue

            if prescript_start:
                if is_similar(line_text, ["*표시가"]):
                    #처방전 품목 종료.
                    prescript_start = False
                    animals_data.append(copy.deepcopy(data))
                    break

                elif is_similar(line_text, ["진료", "백신", "용품", "할인"]): #대분류 항목 및 할인은 제외.
                    continue

                elif not price_pattern.match(line_text.split(' ')[-1]):
                    medicine_name_temp = copy.deepcopy(line_text)
                    is_name_multiline = True
                    medicine_data["bbox"] = [line_bboxes[i]]

                else:
                    name_with_num = ' '.join(line_text.split(' ')[:-1])
                    price = line_text.split(' ')[-1]

                    if is_name_multiline:
                        is_name_multiline = False
                        medicine_name_temp += name_with_num
                        medicine_data["name"] = copy.deepcopy(medicine_name_temp[:-1])
                        medicine_data["bbox"].append(line_bboxes[i])

                    else:
                        medicine_data["name"] = copy.deepcopy(name_with_num[:-1])
                        medicine_data["bbox"].append(line_bboxes[i])

                    if not re.findall(r'\d+', medicine_name_temp):
                        medicine_data["num"] = "1"
                    else:
                        medicine_data["num"] = copy.deepcopy(re.findall(r'\d+', medicine_name_temp)[-1])
                    medicine_data["price"] = copy.deepcopy(price)
                    data["medicines"].append(copy.deepcopy(medicine_data))

        elif receipt_type == 1: #힐링페츠 형식 영수증
            if is_similar(line_text.replace(' ', ''), ["수량금액DC"]):
                prescript_start = True
                data = {"name": [], "medicines": []}
                continue
            
            if prescript_start:
                split_text = line_text.split(' ')
                if is_similar(line_text, ["청구일"]):
                    prescript_start = False
                    break

                if len(split_text) > 2 and price_pattern.match(split_text[-2]):  # DC 가 포함된 가격 line
                    medicine_data["num"] = split_text[0]
                    medicine_data["price"] = split_text[-2]
                    data["medicines"].append(copy.deepcopy(medicine_data))

                elif price_pattern.match(split_text[-1]): #가격이 표시된 line 처리
                    if is_similar(line_text, ["합계", "할인금액"]):
                        pass

                    elif is_similar(line_text, ["부가세포함합계"]):
                        animals_data.append(copy.deepcopy(data))
                        data = {"name": [], "medicines": []}

                    else:
                        medicine_data["num"] = split_text[0]
                        medicine_data["price"] = split_text[-1]
                        medicine_data["bbox"].append(line_bboxes[i])
                        data["medicines"].append(copy.deepcopy(medicine_data))
                        medicine_data = {"name": '', "num": '', "price": '', "bbox": []}

                else:
                    if date_pattern.match(split_text[-1].replace(')', '')):
                        #동물 이름 line 처리
                        name = split_text[0].replace('(', '')
                        name = name[:name.find('[')]

                        data["name"] = [name, line_bboxes[i]]
                    else:
                        medicine_data["name"] = line_text
                        medicine_data["bbox"].append(line_bboxes[i])

    return {"animals": animals_data}


def extract_all_text(dt_boxes, strs, targets):

    if len(dt_boxes) > 0:

        sort_box_mid = []

        for align_bbox, label in zip(dt_boxes, strs):
            x1, y1, x2, y2, x3, y3, x4, y4 = int(align_bbox[0][0]), int(align_bbox[0][1]), \
                                             int(align_bbox[1][0]), int(align_bbox[1][1]), \
                                             int(align_bbox[2][0]), int(align_bbox[2][1]), \
                                             int(align_bbox[3][0]), int(align_bbox[3][1])

            xm, ym = ((x1+x2+x3+x4)/4), ((y1+y2+y3+y4)/4)
            sort_box_mid.append([x1, y1, x3, y3, xm, ym, label])

        # xm 크기로 sorting
        x_sorted_boxes = sorted(sort_box_mid, key=lambda x: np.min(x[4]))

        # 제일 왼쪽부터 나열됨(위아래는 아직 고려x)
        dict_val = {}
        for index, val in enumerate(x_sorted_boxes):
            dict_val[index] = val

        # 영수증 1개 변수 초기화
        skip_idxs = set()
        # smallest_y_overlap = 0.8음
        smallest_y_overlap = 0.5
        lines = {}
        lines_label = []

        # x_sorted_boxes x좌표 좌 -> 우로 한 단어 씩 정렬
        for i in range(len(x_sorted_boxes)):
            if i in skip_idxs:
                continue
            # 가장 우측으로 비교를 통해 하나씩 이동
            rightmost_box_idx = i
            line = [rightmost_box_idx]
            line_key = rightmost_box_idx
            label = [x_sorted_boxes[rightmost_box_idx][6]]
            for j in range(i + 1, len(x_sorted_boxes)):
                if j in skip_idxs:
                    continue
                if is_on_same_line(x_sorted_boxes[rightmost_box_idx][0:4], x_sorted_boxes[j][0:4], smallest_y_overlap):
                    line.append(j)
                    skip_idxs.add(j)
                    rightmost_box_idx = j
                    label.append(x_sorted_boxes[j][6])

            lines_label.append(label)
            lines[line_key] = line

        # 라인단위 첫 글자들만 뽑아서 y위치에 맞춰서 sorting 진행
        first = []
        #print("line keys: ", lines.keys())
        for key_idx in lines.keys():
            # 각 라인의 제일 왼쪽 글자의 값 추가해주시
            first.append((key_idx, x_sorted_boxes[key_idx]))

        # ym기준으로 각 줄의 첫글자 정렬
        y_sorted_boxes = sorted(first, key=lambda first: first[1][5])
        #print("y_sorted_boxes : ", y_sorted_boxes)
        y_sorted_boxes = [y[0] for y in y_sorted_boxes]
        #print("after: y_sorted_boxes : ", y_sorted_boxes)

        new = []
        for y in y_sorted_boxes:
            vlist = lines.get(y)
            new.append(vlist)

        line_by_line = []
        line_bboxes = []
        for line in new:
            tmp_line = []
            max_x = max_y = -1.0
            min_x = min_y = 10000.0

            for val in line:
                tmp_line.append(dict_val[val])
                #print(dict_val[val])
                min_x = dict_val[val][0] if min_x > dict_val[val][0] else min_x
                min_y = dict_val[val][1] if min_y > dict_val[val][1] else min_y
                max_x = dict_val[val][2] if max_x < dict_val[val][2] else max_x
                max_y = dict_val[val][3] if max_y < dict_val[val][3] else max_y

            line_bboxes.append(copy.deepcopy([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]))
            #print([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            line_by_line.append(tmp_line)
            #print("line: ", tmp_line)

        #print("##############################################################################################################################")
        line_texts = []

        for i, line in enumerate(line_by_line):
            line_text = ''
            #print(line)
            for word in line:
                if len(word[6]) == 1:
                    line_text += word[6]
                else:
                    line_text = line_text + ' ' + word[6]
            line_text = line_text.strip()
            line_texts.append(line_text)
            #print(line_text)
        #print("##############################################################################################################################")
        output_json = {}
        output_json['defaults'] = information_finder(line_texts, targets, line_bboxes)

        #print("##############################################################################################################################")

        # 영수증 type, 쿨펫 == 0, 힐링페츠 == 1, etc == -1(error)
        receipt_type = 0
        
        try:
            if not (is_similar(line_texts[0], ['No', 'NO', 'no', 'nO', 'N0', 'N', 'O', 'o'])
                    or is_similar(line_texts[1], ['No', 'NO', 'no', 'nO', 'N0', 'N', 'O', 'o'])): #쿨펫 규격의 영수증여부 확인

                if is_similar(line_texts[1], ['(청구서)', '청구서', '(청구서', '청구서)', '구서', '구서)', '(구서']):
                    # 힐링페츠 규격의 영수증, 2번째 행에 (청구서)가 명시 되어있음.
                    output_json['defaults']['병원명'] = line_texts[0]
                    receipt_type = 1
                elif is_similar(line_texts[0], ['(청구서)', '청구서', '(청구서', '청구서)', '구서', '구서)', '(구서']):
                    output_json['defaults']['병원명'] = '힐링페츠'
                    receipt_type = 1
                elif is_similar(line_texts[4], ['(청구서)', '청구서', '(청구서', '청구서)', '구서', '구서)', '(구서']):
                    output_json['defaults']['병원명'] = line_texts[3]
                    receipt_type = 1
                else:
                    #print('other case')
                    receipt_type = -1
        except:
            receipt_type = -1

        #처방전 부분 결과에 추가.
        output_json["prescript"] = prescript_finder(line_texts, line_bboxes, receipt_type)
        return output_json, receipt_type, line_bboxes, line_texts


    else:
        return {}, -1, [], []


def main():
    start = time.time()

    global_config = config['Global']
    targets = global_config['targets']
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
    # img_name = get_image_file_list(config['Global']['infer_img'])[-1]
    # img_name = './dataset/rec_test/20220812_153552.jpg'
    with open(save_res_path, "w", encoding='utf-8') as fout:
        for img_name in tqdm(get_image_file_list(config['Global']['infer_img'])):
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
            #print(post_result)
            #print(post_result['texts'])
            dt_boxes, strs = post_result['points'], post_result['texts']

            # write resule
            dt_boxes_json = []
            for poly, string in zip(dt_boxes, strs):
                tmp_json = {"transcription": string, 'points': poly.tolist()}
                dt_boxes_json.append(tmp_json)
            otstr = img_name + "\t" + json.dumps(dt_boxes_json, ensure_ascii=False, indent=2) + "\n"
            fout.write(otstr)
            img = cv2.imread(img_name)
            #draw_e2e_res(dt_boxes, strs, config, img, img_name)
            #print(dt_boxes, strs)
            output_json, img_type, line_bboxes, line_texts = extract_all_text(dt_boxes, strs, targets)
            draw_e2e_res(np.array(line_bboxes), line_texts, config, img, img_name)

            save_det_path = os.path.dirname(config['Global']['save_res_path']) + "/random_test/"
            pos_json = []
            for string, box in zip(line_texts, line_bboxes):
                pos_json.append({"text": string, "bbox_pos": box})

            # with open(os.path.join(save_det_path,
            #                        ''.join([os.path.splitext(os.path.basename(img_name))[0], '_bbox', '.json'])), 'w',
            #           encoding='utf-8-sig') as file:
            #     json.dump(pos_json, file, ensure_ascii=False, indent=2)
            '''
            if type != -1:
                print(img_name, '------', output_json)
            else:
                print(img_name, 'is not expected format.')
            '''
            if img_type == -1:
                with open(os.path.join(os.path.dirname(save_res_path), 'random_test', 'other_case.txt'), 'a+',
                          encoding='utf-8') as others:
                    others.write('\n' + img_name)

            with open(os.path.join(os.path.dirname(save_res_path), 'json_result', ''.join([os.path.splitext(os.path.basename(img_name))[0], '.json'])), 'w', encoding='utf-8') as result:
                json.dump(output_json, result, ensure_ascii=False, indent=2)

    logger.info(f"success! running time : {time.time() - start}")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    '''
    target = ['사업자 등록번호', '병원명', '대표자', '고객 번호', '고객 이름', '날짜', '총금액', '청구 금액', '카드 결제', '계좌이체 결제', '결제일']
    '날짜' = 제일 아래의 '결제일'과 동일, 인식률이 아래 결제에 대한 문구가 더 높음(데이터에 더 많은 이유)
    '총금액', '청구금액', '카드결제'는 할인금액이 있는경우 다른 경우가 있음(각 항목을 어떤 용도로 사용할지 몰라서 우선 다 산출)
    '진료내역' 항목은 '동물명'과 '총금액' 사이의 내용을 쭉 긁으면 될텐데, 한글, 영문, 특수기호가 혼재돼있어 인식률 개선이 필요함
    
    만약 영수증이 약간 기울여 촬영됐을 경우, 같은 line에 있는 정보가 같은 line에 있지 않은것처럼 text구성이 될 수 있는데,
    해당 케이스를 잡기 위해서는 바로 위에 있는 정보나 바로 아래에 있는 정보까지도 후보로 올려서
    정규표현식으로 내용을 잡아내야 할 것같
    '''
    main()
