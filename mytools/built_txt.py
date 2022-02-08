#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
# 处理双层目录
def get_imglist(src, img_list):
    for file in os.listdir(src):
        cur_path = os.path.join(src,file)
        if os.path.isdir(cur_path):
            get_imglist(cur_path,img_list)
        else:
            img_list.append(cur_path)
    return img_list

# ********************生成图片名称列表txt文件***************************

# #
# img_path = 'D:/liyiman/datasets/data_chaifen/driver_attr_20211208/yawn/'
# txt_path = 'D:/liyiman/datasets/data_chaifen/driver_attr_20211208/yawn.txt'
# result = []
# file_list = []
# # file_list = get_imglist(img_path, file_list)
# file_list = os.listdir(img_path)
# for file_img in file_list:
#     img_name = os.path.split(file_img)[-1]
#     img_name = 'D:/liyiman/datasets/data_chaifen/driver_attr_20211208/yawn/' + file_img
#     result.append(img_name)
#
# with open(txt_path, 'w') as f:
#     f.write('\n'.join(result))

# *******************生成图片名称+类别标签的txt文件***********************
# import os
# import re
# import numpy as np
#
# img_file_path1 = [
#                   'D:/liyiman/datasets/ZNV_data/face_attr_mask_03/mask/',
#                  ]
# img_file_path2 = [
#                  'D:/liyiman/datasets/ZNV_data/face_attr_mask_03/no_mask/',
#                   ]
# txt_path = 'D:/liyiman/datasets/ZNV_data/face_attr_mask_03/face_attr_mask_03_trainval.txt'
#
# result = []
# file_list2 = []
#
# for file1 in img_file_path1:
#     file_list1 = os.listdir(file1)
#     img_prefix = file1.replace("D:/liyiman/datasets/ZNV_data/", '')
#     for file_img in file_list1:
#         result.append(img_prefix + file_img + ' -1 -1 -1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1')
#
# for f in img_file_path2:
#     file_list = os.listdir(f)
#     img_prefix = f.replace("D:/liyiman/datasets/ZNV_data/", '')
#     for file_img in file_list:
#         result.append(img_prefix + file_img + ' -1 -1 -1 -1 -1 -1 -1 -1 0 -1 -1 -1 -1 -1')
# # 打乱文件列表
# np.random.seed(2)
# np.random.shuffle(result)
# with open(txt_path, 'w') as f:
#     f.write('\n'.join(result))


# **************************修改txt内容************修改条目img_name*************************
txt_path = 'D:/liyiman/datasets/ZNV_data/driver_attr_20211216/val.txt'

def updateFile(file,old_str,new_str):
    """
    替换文件中的字符串
    :param file:文件名
    :param old_str:就字符串
    :param new_str:新字符串
    :return:
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str,new_str)
            # line = 'face_wiki_images/imgs/' + line
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)


# updateFile(txt_path, "cut_ratio5/", "cut_+6/")
updateFile(txt_path, "D:/liyiman/datasets/ZNV_data/driver_attr_20211216/", "")
# updateFile(file_path, "/normal", "normal")

# **************************修改txt中label内容********未知-1——》0*****************************
import os
import re

# txt_path = 'D:/liyiman/datasets/ZNV_data/driver_attr_test/trainval.txt'
def modify_txt(txt_path):
    result = []
    with open(txt_path, 'r', encoding="utf-8") as f:
        # 定义一个用于切割字符串的正则
        seq = re.compile(" ")
        for line in f:
            lst = seq.split(line.strip())
            for i in range(len(lst)):
                if lst[i] == '-1':
                    lst[i] = '0'
            label_str = " ".join(lst)
            result.append(label_str)
    file_path = os.path.splitext(txt_path)[0] + '_new' + os.path.splitext(txt_path)[1]
    with open(file_path, 'w') as f:
        f.write('\n'.join(result))

modify_txt(txt_path)

# **************************修改txt中label内容**根据图片所在分类文件夹修改对应标签类别*****************************
# txt_path = 'D:/liyiman/datasets/Face_attr_mask/train_20211122_new.txt'
# img_path = 'D:/liyiman/datasets/Face_attr_mask/mask_train_0_weizhi/'
#
# def modify_txt_from_folder(txt_path, img_path):
#     img_list = []
#     img_list = get_imglist(img_path, img_list)
#     img_list = [item.replace(img_path, '') for item in img_list]
#     img_list = [item.replace('\\', '/') for item in img_list]
#     # 读取原txt中的条目内容
#     result = []
#     with open(txt_path, 'r', encoding="utf-8") as f:
#         for line in f:
#             result.append(line)
#     # 根据条目是否在对应文件夹中修改对应标签类别
#     seq = re.compile(" ")
#     for i in range(len(result)):
#         lst = seq.split(result[i].strip())
#         if lst[0] in img_list:
#             lst[-6] = '-1'
#             item = " ".join(lst)
#             result[i] = item + '\n'
#     file_path = os.path.splitext(txt_path)[0] + '_new' + os.path.splitext(txt_path)[1]
#     with open(file_path, 'w') as f:
#         f.write(''.join(result))

# modify_txt_from_folder(txt_path, img_path)


# **************************txt文档去重*************************************
# txt_path = 'D:/liyiman/datasets/Face_attr_mask/train.txt'
# new_path = 'D:/liyiman/datasets/Face_attr_mask/train_20211120.txt'
# def quchong_txt(txt_path):
#     img_set = set()
#     result = []
#     with open(txt_path, 'r', encoding="utf-8") as f:
#         seq = re.compile(" ")
#         for line in f:
#             lst = seq.split(line.strip())
#             if lst[0] not in img_set:
#                 img_set.add(lst[0])
#                 result.append(line)
#     with open(new_path, 'w') as f:
#         f.write(''.join(result))
#
# quchong_txt(txt_path)

