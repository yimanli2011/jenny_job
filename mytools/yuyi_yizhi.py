#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import os
import sys


file_sumple = 'D:/liyiman/datasets/shipinchaizhen_1/1_625.json'
file_path = 'D:/liyiman/datasets/cgs/labels/'


def list_files(path):
    files_list = []
    for root, _, files in os.walk(path):
        files_list += files
    return files_list


# 读取样本json文件内容
with open(file_sumple, 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
    if json_data['shapes'][0]['label'] == 'lvdai':
        lvdai_points = json_data['shapes'][0]
    else:
        print('无lvdai类别，请检查文件{}的内容'.format(file_sumple))
        sys.exit()

# 遍历需要加入lvdai_points的json文件列表，逐个加入。
json_files = list_files(file_path)
for file_name in json_files:
    # 逐个读取需要加入lvdai标签的json文件内容
    json_path = file_path + file_name
    exist = False
    with open(json_path, 'r', encoding='utf8')as fp:
        json_data_old = json.load(fp)
        for i in range(len(json_data_old['shapes'])):
            if json_data_old['shapes'][i]['label'] == 'lvdai':
                exist = True
                print('已有lvdai类别，请检查文件{}的内容'.format(file_name))
                break
    if not exist:
        json_data_old['shapes'].insert(0, lvdai_points)
        with open(json_path, "w") as dump_f:
            json.dump(json_data_old, dump_f)
        print("加入文件{}完成...".format(file_name))
