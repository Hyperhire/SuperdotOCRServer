import shutil
import re
import json

label_txt = './pet_receipts/Label.txt'
label_out = './pet_receipts/original_train_label.txt'
shutil.copy(label_txt, '/'.join(re.split('/', label_txt)[:-2]) + '/original_train_label.txt')

with open(label_txt, 'r') as f:
    labels = f.readlines()
with open(label_out, 'r') as f:
    out_labels = f.readlines()

for i, label in enumerate(labels):

    file, label_list = label.split('\t')
    label_point_str = json.loads(label_list)

    tmp_label_point_str = []
    for item in label_point_str:
        item['transcription'] = re.split('@@', item['transcription'])[-1]
        tmp_label_point_str.append(item)

    labels[i] = '\t'.join([file, str(tmp_label_point_str)]) + '\n'

with open(label_txt, 'w') as f:
    for label in labels:
        f.write(label)