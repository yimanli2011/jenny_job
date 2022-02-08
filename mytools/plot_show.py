#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# 可视化feature map上的anchor,以80*80feature map为例
import matplotlib.pyplot as plt
import numpy as np
import torch


def _meshgrid(x, y, row_major=True):
    """Generate mesh grid of x and y.

    Args:
        x (torch.Tensor): Grids of x dimension.
        y (torch.Tensor): Grids of y dimension.
        row_major (bool, optional): Whether to return y grids first.
            Defaults to True.

    Returns:
        tuple[torch.Tensor]: The mesh grids of x and y.
    """
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def single_level_grid_anchors(base_anchors,featmap_size,stride=(16, 16)):
    """Generate grid anchors of a single level.

    Note:
        This function is usually called by method ``self.grid_anchors``.

    Args:
        base_anchors (torch.Tensor): The base anchors of a feature grid.
        featmap_size (tuple[int]): Size of the feature maps.
        stride (tuple[int], optional): Stride of the feature map in order
            (w, h). Defaults to (16, 16).
        device (str, optional): Device the tensor will be put on.
            Defaults to 'cuda'.

    Returns:
        torch.Tensor: Anchors in the overall feature maps.
    """
    feat_h, feat_w = featmap_size
    # convert Tensor to int, so that we can covert to ONNX correctlly
    feat_h = int(feat_h)
    feat_w = int(feat_w)
    shift_x = torch.arange(0, feat_w) * stride[0]
    shift_y = torch.arange(0, feat_h) * stride[1]

    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    shifts = shifts.type_as(base_anchors)
    # first feat_w elements correspond to the first row of shifts
    # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
    # shifted anchors (K, A, 4), reshape to (K*A, 4)

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    all_anchors = all_anchors.view(-1, 4)
    # first A rows correspond to A anchors of (0, 0) in feature map,
    # then (0, 1), (0, 2), ...
    return all_anchors


# base_anchors = torch.Tensor([[ -1.0000,  -2.5000,   9.0000,  10.5000],
#                              [ -4.0000, -11.0000,  12.0000,  19.0000],
#                              [-12.5000,  -7.5000,  20.5000,  15.5000]])
# featmap_size = torch.Size([80, 80])
# stride = (8, 8)
# all_anchors = single_level_grid_anchors(base_anchors, featmap_size, stride)
# list_anchors = all_anchors.numpy().tolist()
#
# ax = plt.gca()
# ax.set_xlim(-20, 120)
# ax.set_ylim(120, -20)
#
# # 网格
# plt.grid(linestyle='-.')  # linestyle = ls
#
# my_x1 = np.arange(-20, 120, 8)
# my_y1 = np.arange(-20, 120, 8)
# plt.xticks(my_x1)
# plt.yticks(my_y1)
#
# # # 画矩形框
# for i, anc in enumerate(list_anchors):
#     if i<3:
#         rect = plt.Rectangle((anc[0], anc[1]), anc[2]-anc[0], anc[3]-anc[1], fill=False, edgecolor='r', linewidth=1)
#         ax.add_patch(rect)
#
# plt.show()
# plt.savefig("D:/liyiman/project_znv/my_test/anchor_80.jpg")

# 定义展示图片的方法：
import matplotlib.pyplot as plt
def show_images(images,index = -1):
    """
    展示并保存图片
    :param images: 需要show的图片
    :param index: 图片名
    :return:
    """
    plt.figure()
    for i, image in enumerate(images):
        ax = plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.imshow(image)
    plt.savefig("data_%d.png"%index)
    plt.show()


# from PIL import Image
# import cv2
# img_file = 'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/aa/03c4e9c9bac53bf16cf71d2ed3bc91d.jpg'
# # im_rr = Image.fromarray(np.uint8('D:/360MoveData/Users/xuhuan/Desktop/accessoryProc_accessoryName=00_64_6402_2_20210823_113145_714_014448234892.jpg')) #img为有问题的图片
# # img = Image.open(img_file)
# img = cv2.imread(img_file)
# # #创建窗口并显示图像
# cv2.namedWindow("Image")
# cv2.imshow("Image",img)
# cv2.waitKey(0)
# #释放窗口
# cv2.destroyAllWindows()

import os
import xml.dom.minidom
import cv2 as cv

ImgPath = 'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/VOC2007/JPEGImages/'
AnnoPath = 'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/VOC2007/Annotations/'  # xml文件地址
save_path = 'D:/liyiman/datasets/ZNV_data/JDWG_Det_09/VOC2007/out'


def draw_anchor(ImgPath, AnnoPath, save_path):
    imagelist = os.listdir(ImgPath)
    os.makedirs(save_path, exist_ok=True)
    for image in imagelist:

        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'
        # print(image)
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 读取图片
        img = cv.imread(imgfile)

        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        print(filename)
        # 得到标签名为object的信息
        objectlist = collection.getElementsByTagName("object")

        for objects in objectlist:
            # 每个object中得到子标签名为name的信息
            namelist = objects.getElementsByTagName('name')
            # 通过此语句得到具体的某个name的值
            objectname = namelist[0].childNodes[0].data

            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
                cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                           thickness=2)
                # cv.imshow('head', img)
                cv.imwrite(save_path + '/' + filename, img)  # save picture

draw_anchor(ImgPath, AnnoPath, save_path)