#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import shutil


# 处理双层目录
# def get_imglist(src, img_list):
#     for file in os.listdir(src):
#         cur_path = os.path.join(src,file)
#         if os.path.isdir(cur_path):
#             get_imglist(cur_path,img_list)
#         else:
#             img_list.append(cur_path)
#     return img_list

path_1 = 'D:/liyiman/datasets/ZNV_data/bileiqi_20220127/labels_sbp'  # 条目不全的文件夹目录
path_2 = 'D:/liyiman/datasets/ZNV_data/bileiqi_20220127/imgs'  # 条目全的文件夹目录
path_3 = 'D:/liyiman/datasets/ZNV_data/bileiqi_20220127/aa'  # 多余条目移动到目标文件夹

# path_1 = r'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/VOC2007/Annotations'  # 条目不全的文件夹目录
# path_2 = r'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/VOC2007/JPEGImages/'  # 条目全的文件夹目录
# path_3 = r'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/VOC2007/aa/'  # 多余条目移动到目标文件夹
#
os.makedirs(path_3, exist_ok=True)
#
list_1 = os.listdir(path_1)
list_2 = os.listdir(path_2)
# list_2 = []
# list_2 = get_imglist(path_2, list_2)

for file_2 in list_2:
    file_1 = file_2.replace(os.path.splitext(file_2)[1], '.json')
    # file_1 = file_2.replace('_pseudo.png', '.json')
    if file_1 not in list_1:
        old_path = os.path.join(path_2, file_2)
        new_path = os.path.join(path_3, file_2)
        # shutil.move(old_path, new_path)
        shutil.copy(old_path, new_path)

# # 查看重复文件数量
# count = 0
# for file_1 in list_1:
#     if file_1 in list_2:
#         count += 1
# print(count)

# 查看重复文件数量,并移出
# count = 0
#
# for file_2 in list_2:
#     if os.path.split(file_2)[1] in list_1:
#         count += 1
#         print(file_2)
#         old_path = file_2
#         new_path = os.path.join(path_3, os.path.split(file_2)[1])
#         shutil.move(old_path, new_path)

# print(count)


# 生成一个文件夹列表并保存为txt文件
# path_file = r'D:/liyiman/datasets/data_chaifen/weigui_300G_81/31011500001310000925'
# txt_path = r'D:/liyiman/datasets/data_chaifen/weigui_300G_81/31011500001310000925.txt'
# list_file = os.listdir(path_file)
# with open(txt_path, 'w') as f:
#     f.write('\n'.join(list_file))


# # 将不在txt文件的图片从file中移动到指定文件夹
# path_file = r'/data/data_01/data2/liyiman/data_raw/weigui_300G_raw/31011500001310000887'
# txt_path = r'/data/data_01/data2/liyiman/data_raw/weigui_300G_raw/31011500001310000887.txt'
# new_file = r'/data/data_01/data2/liyiman/data_raw/weigui_300G_raw/out'
# os.makedirs(new_file, exist_ok=True)
# list_old = os.listdir(path_file)
#
# # 读取原txt中的条目内容
# txt_result = []
# with open(txt_path, 'r', encoding="utf-8") as f:
#     for line in f:
#         txt_result.append(line)
#
# count = 0
# for file_ in list_old:
#     if file_ not in txt_result:
#         count += 1
#         old_path = os.path.join(path_file, file_)
#         new_path = os.path.join(new_file, file_)
#         shutil.move(old_path, new_path)
# print('共移动{}个文件'.format(count))