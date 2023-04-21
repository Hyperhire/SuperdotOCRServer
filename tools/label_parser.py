import re
import json
import glob

root_path = '/home/douglas/PycharmProjects/PPOCRv.2_Android/dataset/영수증_OCR_json/'
files = glob.glob(root_path + '*.json')

make_label = []

for i, file in enumerate(files):
    print(i)
    tmp_output = ""

    with open(file) as f:
        res = json.load(f)
    filename = res.get('name')

    parsed_label = []
    for item in res['labels']:
        tmp_dict = {}

        if item.get('text') == None:
            continue

        tmp_dict['transcription'] = item.get('text', 'NULL')
        vertices = item.get('boundingPoly').get('vertices')
        points = [[round(vertices[0].get('x')), round(vertices[0].get('y'))],
                  [round(vertices[1].get('x')), round(vertices[1].get('y'))],
                  [round(vertices[2].get('x')), round(vertices[2].get('y'))],
                  [round(vertices[3].get('x')), round(vertices[3].get('y'))]]
        tmp_dict["points"] = points
        tmp_dict['difficult'] = False
        tmp_dict['attribute'] = item.get('label')
        tmp_dict['shapeType'] = item.get('boundingPoly').get('type')

        parsed_label.append(tmp_dict)
        parsed_label_json = json.dumps(parsed_label, ensure_ascii=False)

    tmp_output = str(filename) + '\t' + parsed_label_json + '\n'

    make_label.append(tmp_output)


with open(root_path + 'Label.txt', 'w') as ml:
    ml.writelines(make_label)