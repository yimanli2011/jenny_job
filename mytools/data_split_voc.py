#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import random

xmlfilepath = 'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/VOC2007/Annotations'
saveBasePath = 'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/VOC2007/ImageSets/Main/'
os.makedirs(saveBasePath, exist_ok=True)
trainval_percent = 1
train_percent = 0.8
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train size", tr)

ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = os.path.splitext(total_xml[i])[0] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
