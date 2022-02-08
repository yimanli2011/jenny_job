#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import re
import json
import os
import numpy as np
import argparse
import shutil
from tqdm import tqdm


def data_split(files_all):
    ratio_train = 0.85  # 训练集比例
    # ratio_trainval = 0.2  # 验证集比例
    ratio_val = 0.15  # 测试集比例
    assert (ratio_train + ratio_val) == 1.0, 'Total ratio Not equal to 1'  # 检查总比例是否等于1

    cnt_val = round(len(files_all) * ratio_val, 0)
    cnt_train = len(files_all) - cnt_val
    print("test Sample:" + str(cnt_val))
    print("train Sample:" + str(cnt_train))

    # 打乱文件列表
    np.random.seed(2)
    np.random.shuffle(files_all)

    train_list = []
    val_list = []
    trainval_list = []

    for i in range(int(cnt_train)):
        train_list.append(files_all[i])

    for i in range(int(cnt_train), int(cnt_train + cnt_val)):
        val_list.append(files_all[i])

    for i in range(len(files_all)):
        trainval_list.append(files_all[i])

    return train_list, val_list, trainval_list


def txt_split():
    # 文件路径
    save_dir = 'D:/liyiman/datasets/ZNV_data/driver_attr_20211216/'
    img_path = ["D:/liyiman/datasets/ZNV_data/driver_attr_20211216/imgs/"]
    label_path = ["D:/liyiman/datasets/ZNV_data/driver_attr_20211216/labels/"]

    # save_dir = 'D:/liyiman/datasets/ZNV_data/culane_classify_part3/'
    # img_path = ["D:/liyiman/datasets/ZNV_data/culane_classify_part3/imgs/"]
    # label_path = ["D:/liyiman/datasets/ZNV_data/culane_classify_part3/labels/"]

    # save_dir = 'D:/liyiman/datasets/kitti_detection/'
    # img_path = ['D:/liyiman/datasets/kitti_detection/object/training/image_2/training/image_2/']

    # img_path = ["./qixing_imgs_1/", "./qixing_imgs_2/", "./img_done_3/", "./img_done_4/", "./img_gx_done/"]
    # label_path = ["./labels_new_1/", "./labels_new_2/", "./label_done_3/", "./label_done_4/", "./label_gx_done/"]
    result = []
    for i in range(len(img_path)):
        file_list = os.listdir(img_path[i])
        for file_img in file_list:
            file_txt = file_img.replace(os.path.splitext(file_img)[1], '.txt')
            file_txt_path = label_path[i] + file_txt
            # 读取文件
            with open(file_txt_path, 'r', encoding="utf-8") as file:
                # 定义一个用于切割字符串的正则
                seq = re.compile(",")
                # 逐行读取
                for line in file:
                    lst = seq.split(line.strip())
                    image_label = [img_path[i]+file_img] + lst
                    image_label_str = " ".join(image_label)
                    result.append(image_label_str)
    train_list, val_list, trainval_list = data_split(result)
    data_list=[train_list,val_list, trainval_list]
    namels = ['train', 'val', 'trainval']
    for ii in range(len(namels)):
        file_path = save_dir + namels[ii] + '.txt'
        with open(file_path, 'w') as f:
            f.write('\n'.join(data_list[ii]))


def txt_update(img_path, label_path, txt_path, save_path):
    # 1、先将txt中的图片列表读入txt_img_list中
    txt_img_list = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            img_name = line.strip().split(' ')[0].replace('imgs/', '')
            txt_img_list.append(img_name)
    # 2、将在txt_img_list中的img文件夹里的图片及label读取并写入新的txt文件中
    result = []
    for file_img in txt_img_list:
        file_txt = file_img.replace(os.path.splitext(file_img)[1], '.txt')
        file_txt_path = os.path.join(label_path, file_txt)
        # 读取文件
        try:
            with open(file_txt_path, 'r', encoding="utf-8") as file:
                # 定义一个用于切割字符串的正则
                seq = re.compile(",")
                # 逐行读取
                for line in file:
                    lst = seq.split(line.strip())
                    image_label = [os.path.join(img_path, file_img)] + lst
                    image_label_str = " ".join(image_label)
                    result.append(image_label_str)
        except:
            print(file_txt)
    with open(save_path, 'w') as f:
        f.write('\n'.join(result))

def split_by_attr():
    # 文件路径
    attr_num = 8
    save_dir = 'D:/liyiman/datasets/ZNV_data/driver_attr_20211216/target/'
    img_path = ["D:/liyiman/datasets/ZNV_data/driver_attr_20211216/imgs/"]
    label_path = ["D:/liyiman/datasets/ZNV_data/driver_attr_20211216/labels/"]
    os.makedirs(save_dir, exist_ok=True)

    result = []
    for i in range(len(img_path)):
        file_list = os.listdir(img_path[i])
        for file_img in tqdm(file_list, desc='Processing_{}_{}'.format(attr_num, img_path[i])):
            file_txt = file_img.replace(os.path.splitext(file_img)[1], '.txt')
            file_txt_path = label_path[i] + file_txt
            # 读取文件
            with open(file_txt_path, 'r', encoding="utf-8") as file:
                # 定义一个用于切割字符串的正则
                seq = re.compile(" ")
                # 逐行读取
                for line in file:
                    lst = seq.split(line.strip())
                    if lst[attr_num] == '1':
                        old_path = os.path.join(img_path[i], file_img)
                        new_path = os.path.join(save_dir, file_img)
                        shutil.move(old_path, new_path)


if __name__ == '__main__':
    # 生成适配aiw框架的train、val、trainval三个txt 随机分割
    txt_split()

    # 按原有train/val/trainval对应的txt文件中图片列表，更新对应图片的label
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--img_path", default='D:/liyiman/datasets/ZNV_data/driver_attr_08/imgs/', type=str, help='list of image files')
    # parser.add_argument("--label_path", default='D:/liyiman/datasets/ZNV_data/driver_attr_08/labels/',type=str, help='list of label files')
    # parser.add_argument("--txt_path", default='D:/liyiman/datasets/ZNV_data/driver_attr_08/val.txt', type=str, help='txt files for train/val/trainval')
    # parser.add_argument("--new_txt_path", default='D:/liyiman/datasets/ZNV_data/driver_attr_08/val_1104.txt', type=str, help='new txt files for train/val/trainval')
    # args = parser.parse_args()
    #
    # txt_update(args.img_path, args.label_path, args.txt_path, args.new_txt_path)

    # 按属性值提取图片到目标文件夹
    # split_by_attr()