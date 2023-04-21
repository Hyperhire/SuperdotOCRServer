

import os
import time
import string
import argparse
import re
import math
import cv2
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance
import scipy.spatial.distance as distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager, RotatedException, BlurException, get_IoU, get_IoU_inv
from dataset import hierarchical_dataset, AlignCollate, ResizeNormalize, NormalizePAD
from model import Model
from PIL import Image
from operator import itemgetter
from PIL import Image, ImageFont, ImageDraw
import pdb

import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

g_pred_string = []
g_pred_prob = []

# from PIL import ImageFont, ImageDraw, Image
# fontpath = "fonts/gulim.ttc"
# font = ImageFont.truetype(fontpath, 20)


class Recognizer():

    def __init__(self, opt):

        self.opt = opt
        self.right_rotate_checker = 0

        """ vocab / character number configuration """
        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

        """ model configuration """
        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if opt.rgb:
            opt.input_channel = 3

        self.model = Model(opt)
        # self.model = torch.nn.DataParallel(self.model).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        self.model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        self.model = self.model.to(device)
        opt.exp_name = '/'.join(opt.saved_model.split('/')[-4:-1])
        # print(model)

        """ keep evaluation model and result logs """
        # os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
        # os.system(f'{opt.saved_model} ./result/{opt.exp_name}/')

        """ evaluation """
        self.model.eval()

        self.toTensor = transforms.ToTensor()


    def resize_and_concat(self, image, polys, input_channel, filtering):

        resized_images = []
        # croped image 정보는 시각화해서 잘 필터링 됐는지 확인하는 용도, 현재는 사용하지 않음
        # croped_image_list = []
        # rgb_croped_image_list = []
        for poly in polys:
            # image cropping
            croped_image, h, w = self.crop_image(image, poly)
            # rgb_croped_image = croped_image.copy()

            if input_channel == 1 and filtering is None:
                croped_image = self.cvt_color2gray(croped_image)
            elif input_channel == 1 and filtering == 'otsu':
                croped_image, h, w = self.otsu_filtering(croped_image)

            # padding (2 types)
            ##################################################################################################################
            # *** NormalizePAD
            resized_max_w = self.opt.imgW
            transform = NormalizePAD((input_channel, self.opt.imgH, resized_max_w), filtering=filtering)

            ratio = w / float(h)
            if math.ceil(self.opt.imgH * ratio) > self.opt.imgW:
                resized_w = self.opt.imgW
            else:
                resized_w = math.ceil(self.opt.imgH * ratio)
            resized_image = cv2.resize(croped_image, dsize=(resized_w, self.opt.imgH), interpolation=Image.BICUBIC)
            resized_images.append(transform(resized_image))
            # croped_image_list.append(croped_image)
            # rgb_croped_image_list.append(rgb_croped_image)

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        image_tensors = image_tensors.to(device)

        return image_tensors #, croped_image_list, rgb_croped_image_list


    def model_inference(self, image_tensors):

        with torch.no_grad():
            preds = self.model(image_tensors, 'test', is_train=False)  # align with Attention.forward
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, preds_index_list = preds_prob.max(2)
            pred = [''.join([self.converter.character[i] for i in preds_index]) for preds_index in preds_index_list]
            pred_EOS = [p.find('[s]') for p in pred]
            pred_string = [x[:y] for x, y in zip(pred, pred_EOS)]
            pred_max_prob = [x[:y] for x, y in zip(preds_max_prob, pred_EOS)]

            confidence_score = []
            for p in pred_max_prob:
                try:
                    confidence_score.append(p.cumprod(dim=0)[-1].data.tolist())
                except:
                    confidence_score.append(0)

        return pred_string, confidence_score


    def validation(self, image, polys, long_polys, paper_type, filtering=None, save=None):

        if isinstance(save, str):
            if not os.path.isdir(save):
                os.mkdir(save)

        input_channel = 3 if self.opt.rgb else 1

        # 모든 polygon의 좌표, 내용, 확률을 모으는 용도
        self.all_poly_info = []
        self.polys = polys
        self.long_polys = long_polys
        self.paper_type = paper_type

        image_tensors = self.resize_and_concat(image, polys, input_channel, filtering)

        ##########################################
        # Recognizer inference
        ##########################################
        pred_string, confidence_score = self.model_inference(image_tensors)
        global g_pred_string
        g_pred_string = pred_string

        global g_pred_prob
        g_pred_prob = confidence_score
        # print('System output: {}     index: {}     confidence: {}      poly_size: {}'.format(pred_string, i, confidence_score, poly_size))

        for ind in range(len(polys)):
            # print({'index': ind, 'poly': polys[ind], 'pred_string': pred_string[ind], 'confidence_score': confidence_score[ind]})
            # if confidence_score[ind] >= 0.05:
            self.all_poly_info.append({'index': ind, 'poly': polys[ind], 'pred_string': pred_string[ind], 'confidence_score': confidence_score[ind]})
            # else:
            #     self.all_poly_info.append(None)


    def show_result(self, image, polys, long_polys, save=None, filename='test.jpg', filtering=None):

        if isinstance(save, str):
            if not os.path.isdir(save):
                os.mkdir(save)

        input_channel = 3 if self.opt.rgb else 1

        if len(image.shape) == 2:
            temp = Image.fromarray(image, cv2.COLOR_GRAY2RGB)
        else:
            temp = Image.fromarray(image)
        draw = ImageDraw.Draw(temp)

        # 모든 polygon의 좌표, 내용, 확률을 모으는 용도
        self.all_poly_info = []
        self.polys = polys

        image_tensors = self.resize_and_concat(image, polys, input_channel, filtering)

        ##########################################
        # Recognizer inference
        ##########################################
        pred_string, confidence_score = self.model_inference(image_tensors)
        # print('System output: {}     index: {}     confidence: {}      poly_size: {}'.format(pred_string, i, confidence_score, poly_size))

        for ind, poly in enumerate(polys):
            if confidence_score[ind] >= 0.05:
                # self.all_poly_info.append({'index': ind, 'poly': polys[ind], 'pred_string': pred_string[ind], 'confidence_score': confidence_score[ind]})
                # draw.text(tuple(poly[3]), pred_string[ind] + '[' + str(round(confidence_score[ind], 4)) + ']', font=ImageFont.truetype('NanumBarunGothic.ttf', 15),
                #           fill=(0, 50, 100))
                draw.text(tuple(poly[3]), pred_string[ind],
                          font=ImageFont.truetype('NanumBarunGothic.ttf', 15),
                          fill=(0, 50, 100))
                draw.polygon([tuple(poly[0]), tuple(poly[1]), tuple(poly[2]), tuple(poly[3])], outline=(0, 0, 255))
            else:
                # self.all_poly_info.append(None)
                # draw.text(tuple(poly[3]), pred_string[ind] + '[' + str(round(confidence_score[ind], 4)) + ']', font=ImageFont.truetype('NanumBarunGothic.ttf', 15),
                #           fill=(100, 50, 0))
                draw.text(tuple(poly[3]), pred_string[ind],
                          font=ImageFont.truetype('NanumBarunGothic.ttf', 15),
                          fill=(100, 50, 0))
                draw.polygon([tuple(poly[0]), tuple(poly[1]), tuple(poly[2]), tuple(poly[3])], outline=(0, 0, 255))
        # for poly in long_polys:
        #     draw.polygon([tuple(poly[0]), tuple(poly[1]), tuple(poly[2]), tuple(poly[3])], outline=(255, 0, 0))

        if isinstance(save, str):
            result = np.array(temp)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/{}'.format(save, filename), result)
        else:
            # 디버깅용 output 확인
            result = np.array(temp)
            cv2.imshow('OCR', result)
            cv2.waitKey(0)
            cv2.destroyWindow('OCR')


    def show_result_with_textJSON(self, image, output_json, save=None, filename='test.jpg', filtering=None):

        if isinstance(save, str):
            if not os.path.isdir(save):
                os.mkdir(save)

        h, w, c = image.shape
        canvas = Image.new('RGB', (2*w, h), 'white')
        if len(image.shape) == 2:
            temp = Image.fromarray(image, cv2.COLOR_GRAY2RGB)
        else:
            temp = Image.fromarray(image)

        canvas.paste(temp, (0, 0))
        draw = ImageDraw.Draw(canvas, 'RGBA')

        ##########################################
        # Recognizer inference
        ##########################################
        checks = []
        tmp_pred_string = []

        for i, (item, value) in enumerate(output_json.items()):
            checks.append(value['check'])

            if value['check'] == True:
                # print(item, '|||||', value)
                poly = value['tbox_info']['poly']
                # pred_string = str(value['value'])
                pred_string = str(value['tbox_info']['pred_string'])
                if pred_string in tmp_pred_string:
                    draw.text(tuple([w + 10, 10 + 15 * i]), str(item),
                              font=ImageFont.truetype('NanumBarunGothic.ttf', 10),
                              fill=(255, 0, 0))
                    draw.text(tuple([w + 200, 10 + 15 * i]), str(value['value']),
                              font=ImageFont.truetype('NanumBarunGothic.ttf', 10),
                              fill=(255, 0, 0))
                    continue
                tmp_pred_string.append(pred_string)
                draw.text(tuple(poly[3]), pred_string,
                          font=ImageFont.truetype('NanumBarunGothic.ttf', 15),
                          fill=(0, 50, 100))
                draw.polygon([tuple(poly[0]), tuple(poly[1]), tuple(poly[2]), tuple(poly[3])], outline=(0, 255, 0), fill=(0, 255, 0, 100))

                draw.text(tuple([w + 10, 10 + 15 * i]), str(item),
                          font=ImageFont.truetype('NanumBarunGothic.ttf', 10),
                          fill=(255, 0, 0))
                draw.text(tuple([w + 200, 10 + 15 * i]), str(value['value']),
                          font=ImageFont.truetype('NanumBarunGothic.ttf', 10),
                          fill=(255, 0, 0))
            else:
                draw.text(tuple([w + 10, 10 + 15 * i]), str(item),
                          font=ImageFont.truetype('NanumBarunGothic.ttf', 10),
                          fill=(0, 0, 255))
                draw.text(tuple([w + 200, 10 + 15 * i]), str(value['value']),
                          font=ImageFont.truetype('NanumBarunGothic.ttf', 10),
                          fill=(0, 0, 255))

        # canvas.show()

        if not any(checks):
            draw.text(tuple([10, 10]), 'Did not matched',
                      font=ImageFont.truetype('NanumBarunGothic.ttf', 15),
                      fill=(255, 0, 0))

        if isinstance(save, str):
            result = np.array(canvas)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/{}'.format(save, filename), result)
        else:
            # 디버깅용 output 확인
            result = np.array(canvas)
            cv2.imshow('OCR', result)
            cv2.waitKey(0)
            cv2.destroyWindow('OCR')


    def crop_image(self, image, poly):
        rect = cv2.minAreaRect(poly)
        if rect[2] < -45:
            tmp = (rect[0], rect[1][::-1], 0)
            rect = tmp
        box = cv2.boxPoints(rect)
        box = np.int0(box)


        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype('float32')

        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype='float32')

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(image, M, (width, height))

        # # 디버깅용 output 확인
        # cv2.imshow('OCR', warped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return warped, height, width


    def cvt_color2gray(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image


    def otsu_filtering(self, image):
        image = self.cvt_color2gray(image)
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_image = th
        h, w = otsu_image.shape
        return otsu_image, h, w


    def long_align(self, long_polys):

        # poly 정리
        long_polys_lt_rb = np.array(list(map(lambda x: [x[0], x[2]], long_polys)))  # 좌상단, 우하단

        # 좌상단 y값을 기준으로 정렬
        initial_line = np.array([i for i in sorted(enumerate(long_polys_lt_rb), key=lambda x: x[1][0][1])], dtype=object)
        # counting_num = set(range(len(long_polys_lt_rb)))
        counting_num = set(initial_line[:, 0])

        # y값을 기준으로 정렬해보자
        y_points = np.array(list(map(lambda x: np.mean([x[1][0][1], x[1][1][1]]), initial_line)))  # y mean

        final_sorted_list = {}
        line_index = 0

        while True:

            if len(counting_num) == 0:
                break

            initial_value_A = initial_line[min(counting_num)]  # 상단을 기준으로 sorting
            # print(min(counting_num))
            # print(initial_value_A)

            y_mean_point = np.mean([initial_value_A[1][0][1], initial_value_A[1][1][1]])  # initial_value_A의 y축방향 평균
            threshold_value = (y_mean_point - initial_value_A[1][0][1]) * 0.95  # hyper-parameter

            K = list(map(lambda x: [x, abs(x - y_mean_point)], y_points))  # 나머지 polys와 기준 A의 좌상단값과의 y(높이)차이
            K = [[count, i] for count, i in enumerate(K)]
            K = [i for i in K if i[1][1] <= threshold_value]  # 기준 안으로 들어오는 polys / y축 길이도 비슷한지 보자
            sorted_K = list(map(lambda x: x[0], sorted(K, key=lambda x: x[1][1])))  # 먼저 기준점과 y 차이로 sorting하고 결과물 정리
            candidate = list(counting_num.intersection(set(sorted_K)))
            # print(counting_num.intersection(set(sorted_K)))

            # double check
            new_candidate = []
            for cd in candidate:
                new_candidate.append(cd)
                cd_initial_value_A = long_polys_lt_rb[cd]
                cd_y_mean_point = np.mean([cd_initial_value_A[0][1], cd_initial_value_A[1][1]])
                cd_threshold_value = (cd_y_mean_point - cd_initial_value_A[0][1]) * 0.4

                cd_K = list(map(lambda x: [x, abs(x - cd_y_mean_point)], y_points))
                cd_K = [[count, i] for count, i in enumerate(cd_K)]
                cd_K = [i for i in cd_K if i[1][1] <= cd_threshold_value]
                cd_sorted_K = list(map(lambda x: x[0], sorted(cd_K, key=lambda x: x[1][1])))
                # print(cd_sorted_K, cd)
                new_candidate.extend(cd_sorted_K)

            all_candidate = list(counting_num.intersection(set(new_candidate)))
            # print(all_candidate)

            # one_line = []
            # for cd in all_candidate:
            #     half_height = (initial_line[cd][1][1][1] - initial_line[cd][1][0][1]) / 2
            #     tmp_threshold_value = half_height * 0.95  # hyper-parameter
            #
            #     if abs(y_points[cd] - y_mean_point) < tmp_threshold_value:
            #         one_line.append(cd)

            final_sorted_list.update({line_index: list(counting_num.intersection(set(all_candidate)))})
            counting_num = counting_num.difference(counting_num.intersection(set(all_candidate)))
            # print(counting_num)

            line_index += 1

        # print(final_sorted_list)

        return final_sorted_list

    def long_short_mapping(self, long_polys, short_polys):
        # short_polys = recognizer.all_poly_info
        matching_dict = {}
        counting_num = set()
        for ind_l, long in enumerate(long_polys):
            matching_dict.update({ind_l: list()})
            for ind_s, short in enumerate(short_polys):

                if short == None:
                    continue
                long_copy = copy.deepcopy(long)
                # print(ind_l, long_copy, ind_s)
                iou = get_IoU(long_copy, short['poly'])

                if iou > 0.1:
                    iou_inv = get_IoU_inv(long_copy, short['poly'])
                else:
                    iou_inv = 0.0
                short_height = short['poly'][3][1]-short['poly'][0][1]
                long_height = long_copy[3][1] - long_copy[0][1]

                ind = 0.5

                if self.paper_type in ['TTB', 'high_electric']:
                    ind = 0.0

                # print(iou, iou_inv, ind_l, ind_s)
                if iou > 0.75 and short_height > long_height * ind:
                    if short['index'] not in counting_num:
                        tmp_list = matching_dict.get(ind_l)
                        tmp_list.append(short['index']) ##### ind_s
                        matching_dict[ind_l] = tmp_list
                        counting_num.update([short['index']])
                    else:
                        pass

                elif iou_inv > 0.75 and long_height > short_height * ind:
                    if short['index'] not in counting_num:
                        tmp_list = matching_dict.get(ind_l)
                        tmp_list.append(short['index']) ##### ind_s
                        matching_dict[ind_l] = tmp_list
                        counting_num.update([short['index']])
                    else:
                        pass

        return matching_dict


    def line_align(self, long_polys):

        # self.long_polys = long_polys
        final_sorted_list = self.long_align(long_polys)
        # print('final_sorted_list', final_sorted_list)
        self.matching_dict = self.long_short_mapping(long_polys, self.all_poly_info)
        # print('self.matching_dict', self.matching_dict)

        line_and_polys = {}
        for line_num, long_poly_num in final_sorted_list.items():
            short_poly = []
            for long in long_poly_num:
                short_poly.extend(self.matching_dict[long])
                # print('short_poly', short_poly)
            line_and_polys.update({line_num: short_poly})
            # print('line_and_polys', line_and_polys)


        return line_and_polys


    def extract_text(self):

        line_and_polys = self.line_align(self.long_polys)
        final_sorted_text = ''
        final_sorted_text_with_prob = ''
        # k, line = 12, [41, 35, 36, 37, 38, 39, 40]

        for k, line in line_and_polys.items():

            sorted_polys = [i for i in sorted(np.array(self.all_poly_info)[line], key=lambda x: x['poly'][0,0])]  # 거리별로 정렬

            # print(sorted_polys)

            for i, val in enumerate(sorted_polys):
                if i == 0:
                    num_space = max(int(val['poly'][0, 0] / 20), 1)
                    spaces = ' ' * num_space
                    final_sorted_text += spaces
                    final_sorted_text += val['pred_string']

                    final_sorted_text_with_prob += spaces
                    final_sorted_text_with_prob_tmp = str(round(val['confidence_score'], 4))
                    final_sorted_text_with_prob += final_sorted_text_with_prob_tmp
                else:
                    num_space = max(int((val['poly'][0, 0] - sorted_polys[i-1]['poly'][1, 0]) / 20), 1)
                    spaces = ' ' * num_space
                    final_sorted_text += spaces
                    final_sorted_text += val['pred_string']

                    final_sorted_text_with_prob += spaces
                    final_sorted_text_with_prob_tmp = str(round(val['confidence_score'], 4))
                    final_sorted_text_with_prob += final_sorted_text_with_prob_tmp


                if i == len(sorted_polys) - 1:
                    final_sorted_text += '\n'

                    final_sorted_text_with_prob += '\n'

        return final_sorted_text, final_sorted_text_with_prob
