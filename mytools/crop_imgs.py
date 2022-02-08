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
import os.path as osp

from tqdm import tqdm


# img_root = "D:/liyiman/datasets/shida_20210721_1/"
# target_img_root = "D:/liyiman/datasets/shida_20210721_1_new/"
#
# os.makedirs(target_img_root, exist_ok=True)
#
# img_list = [x for x in os.listdir(img_root) if osp.splitext(x)[-1]==".jpg"]
#
# for img_file in tqdm(img_list):
#     base_name = osp.splitext(img_file)[0]
#     print(img_file)
#     new_path = osp.join(target_img_root, img_file)
#     img = cv2.imread(osp.join(img_root,img_file))
#     if img is None:
#         continue
#
#     crop_img = img[50:950, 267:1852]
#     crop_h, crop_w, crop_c = crop_img.shape
#     cv2.imwrite(new_path, crop_img)



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
def pick_rider_points(file_path,label_key):
    # 读取json文件，提取出label为rider的左上点和右下点的坐标。
    with open(file_path, 'r') as f:
        fileJson = json.load(f)
        object_list = []  # 用于存储rider类的points
        count = 0  # 用于记录单张图片中rider类的个数
        shapes = fileJson["shapes"]
        for i in range(len(shapes)):
            if shapes[i]['label'] == label_key:
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
def pick_rider_points_from_xml(file_path,label_key):
    # 读取xml文件，提取出label为rider的左上点和右下点的坐标。
    with open(file_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str.encode('utf-8'))
    label_dict = parse_xml_to_dict(xml)["annotation"]

    object_list = []  # 用于存储rider类的points
    count = 0  # 用于记录单张图片中rider类的个数
    if "object" in label_dict.keys():
        for obj in label_dict["object"]:
            if obj['name'] == label_key:
                # xmin = int(obj["bndbox"]["xmin"])
                # xmax = int(obj["bndbox"]["xmax"])
                # ymin = int(obj["bndbox"]["ymin"])
                # ymax = int(obj["bndbox"]["ymax"])
                xmin = int(float(obj["bndbox"]["xmin"]))
                xmax = int(float(obj["bndbox"]["xmax"]))
                ymin = int(float(obj["bndbox"]["ymin"]))
                ymax = int(float(obj["bndbox"]["ymax"]))
                object_list.append([[xmin, ymin], [xmax, ymax]])
                count += 1
    f.close()
    return object_list, count


def target_vertax_point(clockwise_point):
    #计算顶点的宽度(取最大宽度)
    w1 = np.linalg.norm(clockwise_point[0]-clockwise_point[1])
    w2 = np.linalg.norm(clockwise_point[2]-clockwise_point[3])
    w = w1 if w1 > w2 else w2
    #计算顶点的高度(取最大高度)
    h1 = np.linalg.norm(clockwise_point[1]-clockwise_point[2])
    h2 = np.linalg.norm(clockwise_point[3]-clockwise_point[0])
    h = h1 if h1 > h2 else h2
    #将宽和高转换为整数
    w = int(round(w))
    h = int(round(h))
    #计算变换后目标的顶点坐标
    top_left = [0,0]
    top_right = [w,0]
    bottom_right = [w,h]
    bottom_left = [0,h]
    return np.array([top_left,top_right,bottom_right,bottom_left],dtype=np.float32)


def save_cut_img_4points(img_name, img_path, object_list,nrootdir):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    # img = cv2.imread(img_path)
    if img is None:
        print(img_path)
    width, height = img.shape[1], img.shape[0]
    for j in range(0, len(object_list)):
        # 找出4个点的最小外接矩形裁图
        # rect = cv2.minAreaRect(np.array(object_list[j]))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        # box = list(cv2.boxPoints(rect)) # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点坐标
        # xs = [x[1] for x in box]
        # xs = [math.ceil(x) for x in xs]
        # ys = [x[0] for x in box]
        # ys = [math.ceil(y) for y in ys]
        # cropimage = img[min(xs):max(xs),min(ys):max(ys)]
        # print(cropimage.shape)

        # 按4个点放射变换后的矩形裁图
        points1 = np.float32(object_list[j])
        points2 = target_vertax_point(points1)
        # 计算得到转换矩阵
        M = cv2.getPerspectiveTransform(points1, points2)
        # 实现透视变换转换
        processed = cv2.warpPerspective(img, M, (int(points2[2][0]), int(points2[2][1])))

        if not os.path.isdir(nrootdir):
            os.makedirs(nrootdir)
        cv2.imwrite(nrootdir + '/' + str(img_name) + '_%d.jpg' % j, processed)
        # plt.imshow(processed)
        # plt.show()


def save_cut_img(img_name, img_path, object_list, nrootdir):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    # img = cv2.imread(img_path)
    if img is None:
        print(img_path)
    width, height = img.shape[1], img.shape[0]
    for j in range(0, len(object_list)):
        """
        轮廓的切割主要是通过数组切片实现的，不过这里有一个小技巧：就是图片切割的w, h是宽和高，而数组讲的是行（row）和列（column）。
        所以，在切割图片时，数组的高和宽是反过来写的。
        """
        # 按一定比例将边界外扩
        w_ = round((object_list[j][1][0]-object_list[j][0][0])*0.25/2)
        h_ = round((object_list[j][1][1]-object_list[j][0][1])*0.25/2)
        x1 = object_list[j][0][0] - w_
        x2 = object_list[j][1][0] + w_
        y1 = object_list[j][0][1] - h_
        y2 = object_list[j][1][1] + h_
        # 直接外扩一定尺寸
        # x1 = object_list[j][0][0] - 3
        # x2 = object_list[j][1][0] + 3
        # y1 = object_list[j][0][1] - 3
        # y2 = object_list[j][1][1] + 3
        # 判断是否出界
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


def save_cut_img_by_pix(img_name, img_path, nrootdir):
    if not os.path.isdir(nrootdir):
        os.makedirs(nrootdir)
    img = cv2.imdecode(np.fromfile(osp.join(img_path, img_name), dtype=np.uint8), -1)
    if img is None:
        print(img_path)
    width, height = img.shape[1], img.shape[0]
    # 先用y确定高，再用x确定宽  y1: y2  , x1 : x2
    newimage = img[115:725, 0:width]
    cv2.imwrite(osp.join(nrootdir, img_name), newimage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', default='/data/data_01/liukai/data/special_vehicle/Annotations_beifen', type=str, help='xml file dir')
    parser.add_argument('--img_dir', default='/data/data_01/liukai/data/special_vehicle/JPEGImages', type=str, help='img file dir')
    parser.add_argument('--cut_img_dir', default='/data/data_01/liukai/data/special_vehicle/cut_gongchengche', type=str, help='cut_img file dir')
    parser.add_argument('--label_key', default='工程车', type=str, help='label key')
    args = parser.parse_args()
    start = time.time()
    # 先修改文件夹路径
    label_path = args.xml_dir
    # label_path = 'xml'
    img_path = args.img_dir
    label_key = args.label_key
    cut_img_dir = args.cut_img_dir
    os.makedirs(cut_img_dir, exist_ok=True)
    file_list = os.listdir(label_path)
    img_list = os.listdir(img_path)
    count_sum = 0  # 用于记录截取的图片数
    for i in tqdm(file_list):
        file_path = label_path + '/' + i
        # 获得所需label框的坐标点列表和个数
        # object_list, count = pick_rider_points(file_path,label_key)
        object_list, count = pick_rider_points_from_xml(file_path,label_key)
        im = osp.splitext(i)[0]
        im_jpg = '%s.jpg' % (im)
        im_png = '%s.png' % (im)
        im_jpeg = '%s.jpeg' % (im)
        if im_jpg in img_list:
            im_path_rs = osp.join(img_path, im_jpg)
        elif im_png in img_list:
            im_path_rs = osp.join(img_path, im_png)
        elif im_jpeg in img_list:
            im_path_rs = osp.join(img_path, im_jpeg)
        # 根据坐标点，将图片中的相应位置裁剪并保存
        # save_cut_img_4points(im, im_path_rs, object_list,cut_img_dir)
        save_cut_img(im, im_path_rs, object_list,cut_img_dir)
        # print("图片%s" % im + '共有%d个' % count + label_key)
        count_sum += count
    print('共生成%d张' % count_sum + '%s图片' % label_key)
    print('cost time:' + str(time.time()-start))

    # # 按一定像素值裁剪图片
    # img_path = args.img_dir
    # cut_img_dir = args.cut_img_dir
    # img_list = os.listdir(img_path)
    # for img in tqdm(img_list):
    #     save_cut_img_by_pix(img, img_path, cut_img_dir)