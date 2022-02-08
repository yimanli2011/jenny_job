#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import os
import math
import cv2
import time
from lxml import etree
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# from json
def pick_rider_points(file_path):
    # 读取json文件，提取出label为rider的左上点和右下点的坐标。
    with open(file_path, 'r') as f:
        fileJson = json.load(f)
        object_list = []  # 用于存储rider类的points
        count = 0  # 用于记录单张图片中rider类的个数
        shapes = fileJson["shapes"]
        for i in range(len(shapes)):
            if shapes[i]['label'] == 'rider':
                points = shapes[i]["points"]
                new_points = []
                for point in points:
                    point = [math.ceil(xy) for xy in point]  # 向上取整
                    new_points.append(point)
                object_list.append(new_points)
                count += 1
    f.close()
    return object_list, count

# from xml
def pick_rider_points_from_xml(file_path):
    # 读取xml文件，提取出label为rider的左上点和右下点的坐标。
    with open(file_path, 'r') as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str)
    label_dict = parse_xml_to_dict(xml)["annotation"]

    object_list = []  # 用于存储rider类的points
    count = 0  # 用于记录单张图片中rider类的个数
    if "object" in label_dict.keys():
        for obj in label_dict["object"]:
            if obj['name'] == 'rider':
                xmin = int(obj["bndbox"]["xmin"])
                xmax = int(obj["bndbox"]["xmax"])
                ymin = int(obj["bndbox"]["ymin"])
                ymax = int(obj["bndbox"]["ymax"])
                object_list.append([[xmin, ymin], [xmax, ymax]])
                count += 1
    f.close()
    return object_list, count


def save_cut_img(img_name, img_path, object_list,nrootdir):
    # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.imread(img_path)
    width, height = img.shape[1], img.shape[0]
    for j in range(0, len(object_list)):
        """
        轮廓的切割主要是通过数组切片实现的，不过这里有一个小技巧：就是图片切割的w, h是宽和高，而数组讲的是行（row）和列（column）。
        所以，在切割图片时，数组的高和宽是反过来写的。
        """
        # 判断是否出界
        y1 = object_list[j][0][1] - 3
        y2 = object_list[j][1][1] + 3
        x1 = object_list[j][0][0] - 3
        x2 = object_list[j][1][0] + 3
        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0
        if y2 > height:
            y2 = height
        if x2 > width:
            x2 = width
        # 先用y确定高，再用x确定宽  y1: y2  , x1 : x2
        newimage = img[y1:y2, x1:x2]
        if not os.path.isdir(nrootdir):
            os.makedirs(nrootdir)
        cv2.imwrite(nrootdir + '/' +str(img_name) + '_%d.jpg' % j, newimage)
        # plt.imshow(newimage)
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', default='D:/liyiman/datasets/rider_raw/xml_done', type=str, help='json file dir')
    parser.add_argument('--img_dir', default='D:/liyiman/datasets/rider_raw/img_done', type=str, help='img file dir')
    parser.add_argument('--cut_img_dir', default='D:/liyiman/datasets/rider_raw/cut_image', type=str, help='cut_img file dir')

    args = parser.parse_args()
    start = time.time()
    # 先修改文件夹路径
    label_path = args.json_dir
    # label_path = 'xml'
    img_path = args.img_dir
    cut_img_dir = args.cut_img_dir
    file_list = os.listdir(label_path)
    count_sum = 0  # 用于记录截取的rider图片数
    for i in file_list:
        file_path = label_path + '/' + i
        # 获得所需label框的坐标点列表和个数
        # object_list, count = pick_rider_points(file_path)
        object_list, count = pick_rider_points_from_xml(file_path)
        im = i.split('.xml')[0]
        im_path = img_path + '/%s.jpg' % (im)
        # 根据坐标点，将图片中的相应位置裁剪并保存
        save_cut_img(im, im_path, object_list,cut_img_dir)
        print("图片%s" % im + '共有%d个rider' % count)
        count_sum += count
    print('共生成%d张rider图片' % count_sum)
    print('cost time:' + str(time.time()-start))