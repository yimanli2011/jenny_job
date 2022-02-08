import xml.etree.ElementTree as ET
import pickle
import os, sys
from os import listdir, getcwd
from os.path import join

classes = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'rider', 'tricycle']
# classes = ['smoking', 'calling', 'playing']


def convert(size, box):
    # Conversion from VOC to YOLO format
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# print ann_file
def converting_annotation(ann_dir, save_path):
    for ann in os.listdir(ann_dir):
        txt_file = ann[:-4] + '.txt'
        in_file = open(os.path.join(ann_dir, ann))
        out_file = open(os.path.join(save_path, txt_file), 'w')
        try:
            tree = ET.parse(in_file)
        except:
            print(ann)
            continue
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if w == 0 or h == 0:
            continue
        
        for obj in root.iter('object'):
            name = obj.find('name').text.strip().lower()
            # if name in ('car'):
            #     name = 'car'
            # elif name in ('bus'):
            #     name = 'bus'
            # elif name in ('pedestrian', 'pedestrian(sitting)', 'people', 'person'):
            #     name = 'person'
            # elif name in ('bike', 'bicycle'):
            #     name = 'bike'
            # elif name in ('truck'):
            #     name = 'truck'
            # elif name in ('motor', 'motorcycle', 'motorbike'):
            #     name = 'motor'
            # elif name in ('rider', 'cyclist'):
            #     name = 'rider'
            # elif name in ('tricycle', 'awning-tricycle'):
            #     name = 'tricycle'
            # else:
            #     continue
            if name not in classes:
                continue
            cls_id = classes.index(name)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            data = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in data]) + '\n')


if __name__ == '__main__':
    ann_dir = 'D:/liyiman/datasets/ZNV_data/rfc_chepai_qingdao/VOC2007/Annotations_new'
    
    converting_annotation(ann_dir, os.path.join(os.path.split(ann_dir)[0], 'yolo_anno_new'))
