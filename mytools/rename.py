#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import shutil
from tqdm import tqdm

# 处理双层目录
def get_imglist(src, img_list):
    for file in os.listdir(src):
        cur_path = os.path.join(src,file)
        if os.path.isdir(cur_path):
            get_imglist(cur_path,img_list)
        else:
            img_list.append(cur_path)
    return img_list


def rename_subfolders():
    for foldName, subfolders, filenames in os.walk(path):     #用os.walk方法取得path路径下的文件夹路径，子文件夹名，所有文件名
        for i in tqdm(range(len(filenames))):
            new_name = 'weigui_hp_20220130_' +'%05d' % (i + 1) + ".jpg"
            # new_name = filename.replace(os.path.splitext(img_first)[1], '')
            # new_name = filenames[i].replace('微信图片_', '')
            # new_name = filenames[i]+'.jpg'
            os.rename(os.path.join(foldName, filenames[i]), os.path.join(foldName, new_name))  #子文件夹重命名


def repath(path, new_dir):
    os.makedirs(new_dir, exist_ok=True)
    img_list = []
    img_list = get_imglist(path, img_list)

    for img in img_list:
        new_name = img.replace('\\', '_')
        old_path = img
        new_path = os.path.join(new_dir, os.path.split(new_name)[1])
        shutil.move(old_path, new_path)


if __name__ == '__main__':
    path = r'D:\liyiman\datasets\data_chaifen\weigui_huangpu_20220130'   #运行程序前，记得修改主文件夹路径！
    rename_subfolders()          #调用定义的函数，注意名称与定义的函数名一致

    # new_path = 'D:/liyiman/datasets/Face_attr_mask/raw/useful_data/AFDB_masked_face/'
    # repath(path, new_path)     # 处理双层目录，将双层目录文件变为单层目录，并对各个文件重命名