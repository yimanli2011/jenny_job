#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# -*- coding: utf-8 -*-
# !/usr/bin/env python
# @Desc    : 图片的hash算法

import cv2
import os
import argparse
import shutil
import imghdr
import numpy as np
import hashlib
import json
from tqdm import tqdm


# 均值哈希算法
def ahash(image):
    # 将图片缩放为8*8的
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # s为像素和初始灰度值，hash_str为哈希值初始值
    s = 0
    ahash_str = ''
    # 遍历像素累加和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 计算像素平均值
    avg = s / 64
    # 灰度大于平均值为1相反为0，得到图片的平均哈希值，此时得到的hash值为64位的01字符串
    ahash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                ahash_str = ahash_str + '1'
            else:
                ahash_str = ahash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(ahash_str[i: i + 4], 2))
    # print("ahash值：",result)
    return result


# 差异值哈希算法
def dhash(image):
    # 将图片转化为8*8
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result


# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def readtxt(path):
    files_list = []
    label_list = []
    dir_path = os.path.dirname(path)
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            img_path = line.strip().split(' ')[0]
            file_path = dir_path + '/' + img_path
            files_list.append(file_path)
            label = line.strip().split(' ')[1:]
            label_list.append([int(i) for i in label])
    return files_list, label_list


def process_md5(args):
    # 获取图片列表
    img2_list = []
    count = 0
    same_list = []

    for i in range(len(args.img2_file)):
        if os.path.isdir(args.img2_file[i]):
            img2_ = os.listdir(args.img2_file[i])
            img2_ = [os.path.join(args.img2_file[i], j) for j in img2_]
            img2_list += img2_
        else:
            img2_, _ = readtxt(args.img2_file[i])
            img2_list += img2_
    temp_dict = {}

    for item2 in tqdm(img2_list, desc='Processing img2_file'):
        img2 = cv2.imread(item2)
        if img2 is None:
            print("异常图片：{}".format(item2))
        else:
            img_array = np.array(img2)  # 转为数组
            md5 = hashlib.md5()  # 创建一个hash对象
            md5.update(img_array)  # 获得当前文件的md5码
            if md5.hexdigest() not in temp_dict.keys():  # 如果当前的hash码不在集合中
                temp_dict[md5.hexdigest()] = item2  # 则添加当前hash码到集合中
            else:
                count += 1
                same_list.append(item2)
    print('共有相同图片{}张'.format(count))
    # 如果指定文件夹，则将相同图片移动到指定文件夹
    if args.imgs_outpath != '':
        os.makedirs(args.imgs_outpath, exist_ok=True)
        for same_one in tqdm(same_list, desc='Moving same imgs'):
            old_path = same_one
            new_path = os.path.join(args.imgs_outpath, os.path.split(same_one)[1])
            shutil.move(old_path, new_path)
    else:
        for same_one in tqdm(same_list, desc='Moving same imgs'):
            old_path = same_one
            new_dir = os.path.split(same_one)[0] + '_quchong'
            os.makedirs(new_dir, exist_ok=True)
            new_path = os.path.join(new_dir, os.path.split(same_one)[1])
            shutil.move(old_path, new_path)

    item = json.dumps(temp_dict, indent=4)
    if not os.path.exists(args.json_save_path):
        with open(args.json_save_path, "w", encoding='utf-8') as f:
            f.write(item)
            print("^_^ write success")
    else:
        with open(args.json_save_path, "a", encoding='utf-8') as f:
            f.write(item)
            print("^_^ write success")


def md5_quchong(args):
    count = 0
    temp_list = []
    os.makedirs(args.imgs_outpath, exist_ok=True)
    # 读取字典
    md5_dic = json.load(open(args.json_save_path, 'r'))
    # 判断img1_file是文件夹还是txt文件
    # 如果是文件夹，则直接去重，将相同图片移动到指定文件夹
    if os.path.isdir(args.img1_file):
        img1_list = os.listdir(args.img1_file)
        for item_i in tqdm(range(len(img1_list)), desc='Processing img1_file'):
            if imghdr.what(os.path.join(args.img1_file,img1_list[item_i])):  # 判断是否能打开
                img1 = cv2.imread(os.path.join(args.img1_file,img1_list[item_i]))
                if img1 is None:
                    print("异常图片：{}".format(img1_list[item_i]))
                else:
                    img_array = np.array(img1)  # 转为数组
                    md5 = hashlib.md5()  # 创建一个hash对象
                    md5.update(img_array)  # 获得当前文件的md5码
                    if md5.hexdigest() in md5_dic.keys():
                        print('相同图片为：')
                        print(md5_dic[md5.hexdigest()])
                        print(img1_list[item_i])
                        count += 1
                        old_path = os.path.join(args.img1_file,img1_list[item_i])
                        new_path = os.path.join(args.imgs_outpath, img1_list[item_i])
                        shutil.move(old_path, new_path)
        print('共有相同图片{}张'.format(count))
    # 如果是txt文件，则将去重后的图片及标签，重新组合写入新的txt文件。
    else:
        img1_list, label_list = readtxt(args.img1_file)
        for item_i in tqdm(range(len(img1_list)), desc='Processing img1_file'):
            if imghdr.what(img1_list[item_i]):  # 判断是否能打开
                img1 = cv2.imread(img1_list[item_i])
                if img1 is None:
                    print("异常图片：{}".format(img1_list[item_i]))
                else:
                    img_array = np.array(img1)  # 转为数组
                    md5 = hashlib.md5()  # 创建一个hash对象
                    md5.update(img_array)  # 获得当前文件的md5码
                    if md5.hexdigest() in md5_dic.keys():
                        print('相同图片为：')
                        print(md5_dic[md5.hexdigest()])
                        print(img1_list[item_i])
                        count += 1
                    else:
                        temp_list.append(img1_list[item_i] + " " + " ".join([str(i) for i in label_list[item_i]]))
        # 生成新的txt文件
        file_path = os.path.splitext(args.img1_file)[0] + '_new' + os.path.splitext(args.img1_file)[1]
        with open(file_path, 'w') as f:
            f.write('\n'.join(temp_list))
            print("^_^ write success")
        print('共有相同图片{}张'.format(count))


def process_ahash(args):
    os.makedirs(args.imgs_outpath, exist_ok=True)
    # 获取图片列表
    img2_list = []
    img1_list, _ = readtxt(args.img1_file)
    for i in range(len(args.img2_file)):
        img2_, _ = readtxt(args.img2_file[i])
        img2_list += img2_
    temp_dict = {}

    for item2 in img2_list:
        img2 = cv2.imread(item2)
        if img2 is None:
            print("异常图片：{}".format(item2))
        else:
            hash2_a = ahash(img2)  # 获得当前文件的哈希值
        if hash2_a not in temp_dict.keys():  # 如果当前的hash码不在集合中
            temp_dict[hash2_a] = item2  # 则添加当前hash码到集合中

    item = json.dumps(temp_dict, indent=4)
    if not os.path.exists(args.json_save_path):
        with open(args.json_save_path, "w", encoding='utf-8') as f:
            f.write(item)
            print("^_^ write success")
    else:
        with open(args.json_save_path, "a", encoding='utf-8') as f:
            f.write(item)
            print("^_^ write success")

    for item1 in img1_list:
        if imghdr.what(item1):  # 判断是否能打开
            img1 = cv2.imread(item1)
            if img1 is None:
                print("异常图片：{}".format(item1))
            else:
                hash1_a = ahash(img1)  # 获得当前文件的哈希值
            if hash1_a in temp_dict.keys():
                print('相似图片为：')
                print(temp_dict[hash1_a])
                print(item1)
                old_path = item1
                new_path = os.path.join(args.imgs_outpath, os.path.split(item1)[1])
                shutil.move(old_path, new_path)

    # camphash1 = campHash(hash1_a, hash2_a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='some configs for automatic operation')
    parser.add_argument('--img1_file', default='D:/liyiman/datasets/ZNV_data/face_attr_mask_03/mask',
                        help='test img file path,txt or file')
    parser.add_argument('--img2_file',
                        default=[
                                 r'D:\liyiman\datasets\data_chaifen\weigui_huangpu_20220113',
                                 # 'D:/liyiman/datasets/Face_attr_mask/person_mask_test/JPEGImages/',
                                 # 'D:/liyiman/datasets/Face_attr_mask/facemask/JPEGImages',
                                 # 'D:/liyiman/datasets/Face_attr_mask/maskDetect/maskimages',
                                 # 'D:/liyiman/datasets/Face_attr_mask/voc/JPEGImages',
                                 # 'D:/liyiman/datasets/ZNV_data/face_attr_mask_03/mask',
                                 # 'D:/liyiman/datasets/ZNV_data/face_attr_mask_03/no_mask',
                                 # 'D:/liyiman/datasets/Face_attr_mask/raw/useful_data/BaiDuCrawler001Img',
                                 # 'D:/liyiman/datasets/Face_attr_mask/raw/useful_data/kitchen08_04Img/',
                                 # 'D:/liyiman/datasets/Face_attr_mask/raw/useful_data/AFDB_masked_face',
                                 # 'D:/liyiman/datasets/data_chaifen/driver_attr_20211216/safebelt/',
                                 ],
                        help='train img file path')
    # D:/liyiman/datasets/ZNV_data/face_attr_mask_01/md5_dict_face01.json
    # D:/liyiman/datasets/Face_attr_mask/test_md5_dict.json
    parser.add_argument('--imgs_outpath', default=r'D:\liyiman\datasets\data_chaifen\weigui_huangpu_20220113_quchong', help='')
    parser.add_argument('--json_save_path', default=r'D:\liyiman\datasets\data_chaifen\weigui_huangpu_20220113_aa.json', help='')
    args = parser.parse_args()

    # 找出测试集中与训练集中相同的图片，并将该图片从测试集中移至目标文件夹
    process_md5(args)  # 计算并保存img2_file的文件中图片的md5值字典{MD5:图片}
    # md5_quchong(args)  # 逐一计算img1_file中图片的md5值，判断是否在以上字典中，在则认为是相同图片，并做处理
    # process_ahash(args)  # 找出相似的图片
