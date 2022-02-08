#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import division
import os
import xml.dom.minidom
import cv2
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def label_img(ImgPath,LableName,Savepath):
    imagelist = os.listdir(ImgPath)
    for image in imagelist:
        imgfile = os.path.join(ImgPath, image)
        im = cv2.imread(imgfile)
        h, w, _ = im.shape
        if not os.path.exists(Savepath):
            os.makedirs(Savepath)
        path = os.path.join(Savepath, image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(im, LableName, (100, 100), font, 1, (0, 0, 255), 1)
        img_out = cv2ImgAddText(im, LableName, 10, 65, (238, 238, 0), 20)
        # print(path)
        cv2.imwrite(path, img_out)

def read_xml(ImgPath, AnnoPath, Savepath):
    imagelist = os.listdir(AnnoPath)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        # imgfile =  +'/'+ image_pre+ '.JPG'
        imgfile = os.path.join(ImgPath, image_pre + '.jpg')
        # xmlfile = AnnoPath +'/'+ image_pre+ '.xml'
        xmlfile = os.path.join(AnnoPath, image_pre + '.xml')
        # print(imgfile)
        # print(xmlfile)
        im = cv2.imread(imgfile)
        # im = cv2.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # imdecode()读取图像数据并转换成图片格式
        # fromfile()读数据时需要用户指定元素类型，并对数组的形状进行适当的修改，cv2.IMREAD_UNCHANGED加载图像
        DomTree = xml.dom.minidom.parse(xmlfile)  # 读取xml文件中的值
        annotation = DomTree.documentElement  # documentElement 属性可返回文档的根节点。
        filenamelist = annotation.getElementsByTagName(
            'filename')  # getElementById()可以访问Documnent中的某一特定元素，顾名思义，就是通过ID来取得元素，所以只能访问设置了ID的元素。
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        i = 1
        for objects in objectlist:
            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data  # 通过xml文件给图像加目标框
            bndbox = objects.getElementsByTagName('bndbox')
            if objectname == "playing" or objectname == "smoking" or objectname == "calling":
                for box in bndbox:
                    try:
                        x1_list = box.getElementsByTagName('xmin')
                        x1 = int(x1_list[0].childNodes[0].data)

                        y1_list = box.getElementsByTagName('ymin')
                        y1 = int(y1_list[0].childNodes[0].data)

                        x2_list = box.getElementsByTagName('xmax')
                        x2 = int(x2_list[0].childNodes[0].data)

                        y2_list = box.getElementsByTagName('ymax')
                        y2 = int(y2_list[0].childNodes[0].data)

                        minX = x1
                        minY = y1
                        maxX = x2
                        maxY = y2

                        # if objectname == "chache":
                        color = (255, 0, 80)
                        # elif objectname == "boots":
                        #     color = (255, 0, 255)
                        # elif objectname == "glove":
                        #     color = (51, 204, 204)
                        # elif objectname == "oversleeves":
                        #     color = (255, 0, 0)
                        # else:
                        #     color = (255, 0, 0)

                        cv2.rectangle(im, (minX, minY), (maxX, maxY), color, 2)

                        if not os.path.exists(Savepath):
                            os.makedirs(Savepath)
                        path = os.path.join(Savepath, image_pre + '.jpg')
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(im, objectname, (minX, minY - 7), font, 1, (0, 0, 255), 2)
                        # print(path)
                        # cv2.imwrite(path, im)
                        cv2.imencode(".jpg", im)[1].tofile(path)
                        i += 1
                    except Exception as e:
                        print(e)


if __name__ == "__main__":
    img_path = r'D:/liyiman/aa/ztc_img/'
    xml_path = r'D:/liyiman/aa/ztc_ll/'
    save_path = r'D:/liyiman/aa/save/'
    lablename = '正常'
    read_xml(img_path, xml_path, save_path)
    # label_img(img_path,lablename,save_path)
