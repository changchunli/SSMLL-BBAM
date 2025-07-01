import os
import json
import numpy as np
import pandas as pd
import argparse
import string

pp = argparse.ArgumentParser(description='Format VOC 2012 metadata.')
pp.add_argument('--load_path', type=str, default='Dataset/Animals_with_Attributes2', help='Path to a directory containing a copy of the VOC dataset.')
pp.add_argument('--save_path', type=str, default='Dataset/Animals_with_Attributes2', help='Path to output directory.')
args = pp.parse_args()


all_label = open(args.load_path+'/predicate-matrix-binary.txt')
label_idx = all_label.readlines()

all_class = open(args.load_path+'/classes.txt')
class_idx = all_class.readlines()
class_all = []
for i in class_idx:
    class_all.append(i.replace('\n', '').replace('\t', '').strip().strip(string.digits))


image_list = {'train': {}, 'test': {}}

for phase in ['train', 'test']:
    class_file = open(os.path.join(args.load_path, phase + 'classes.txt'))
    class_path = class_file.readlines()
    label_matrix = []

    for classes in class_path:
        name = classes.strip('\n')
        images = os.listdir(os.path.join(args.load_path, 'JPEGImages', name))
        label_id = class_all.index(name)

        for i in images:
            image_list[phase][i] = [int(x) for x in label_idx[label_id].strip('\n').split()]

    img = []
    label_matrix = []
    for i in image_list[phase]:
        img.append(i)
        label_matrix.append(image_list[phase][i])
   
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels.npy'), np.array(label_matrix))
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images.npy'), np.array(img))

