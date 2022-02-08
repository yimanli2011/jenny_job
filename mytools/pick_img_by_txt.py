#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import shutil

# 将不在txt文件的图片从file中移动到指定文件夹
path_file = r'D:/liyiman/datasets/data_chaifen/weigui_300G_81/31011500001310000887'
txt_path = r'D:/liyiman/datasets/data_chaifen/weigui_300G_81/31011500001310000887.txt'
new_file = r'D:/liyiman/datasets/data_chaifen/weigui_300G_81/out'
os.makedirs(new_file, exist_ok=True)
list_old = os.listdir(path_file)

# 读取原txt中的条目内容
txt_result = []
with open(txt_path, 'r', encoding="utf-8") as f:
    for line in f:
        txt_result.append(line.strip())

count = 0
for file_ in list_old:
    if file_ not in txt_result:
        count += 1
        old_path = os.path.join(path_file, file_)
        new_path = os.path.join(new_file, file_)
        shutil.move(old_path, new_path)
print('共移动{}个文件'.format(count))