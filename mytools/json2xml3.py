#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import xml.dom
import xml.dom.minidom
import os
# from PIL import Image
import cv2
import json

_IMAGE_PATH = 'E:/coco/COCO/train'

_INDENT = '' * 4
_NEW_LINE = '\n'
_FOLDER_NODE = 'ShanghaiDataSet'
_ROOT_NODE = 'annotation'
_DATABASE_NAME = ''
_ANNOTATION = 'COCO2014'
_AUTHOR = 'li'
_SEGMENTED = '0'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'

# _IMAGE_COPY_PATH= 'JPEGImages'
_ANNOTATION_SAVE_PATH = 'D:/liyiman/project_znv/mytool/jsonToXml/xmlDataSet/Annotations'


# 封装创建节点的过程
def createElementNode(doc, tag, attr):  # 创建一个元素节点
    element_node = doc.createElement(tag)

    # 创建一个文本节点
    text_node = doc.createTextNode(attr)

    # 将文本节点作为元素节点的子节点
    element_node.appendChild(text_node)

    return element_node


def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)

    parent_node.appendChild(child_node)


# object节点比较特殊
def createObjectNode(doc, attrs):
    object_node = doc.createElement('object')
    # print("创建object中")
    midname = attrs["label_name"]

    createChildNode(doc, 'name', midname,
                    object_node)
    createChildNode(doc, 'pose',
                    _POSE, object_node)
    createChildNode(doc, 'truncated',
                    _TRUNCATED, object_node)
    createChildNode(doc, 'difficult',
                    _DIFFICULT, object_node)
    bndbox_node = doc.createElement('bndbox')
    # print("midname1[points]:",midname1["points"])
    # 比较points大小确定左上角、右下角点
    if attrs['bndbox']['xmin'] > attrs['bndbox']['xmax']:
        temp1 = attrs['bndbox']['xmin']
        attrs['bndbox']['xmin'] = attrs['bndbox']['xmax']
        attrs['bndbox']['xmax'] = temp1
    if attrs['bndbox']['ymin'] > attrs['bndbox']['ymax']:
        temp2 = attrs['bndbox']['ymin']
        attrs['bndbox']['ymin'] = attrs['bndbox']['ymax']
        attrs['bndbox']['ymax'] = temp2
    createChildNode(doc, 'xmin', str(int(attrs['bndbox']['xmin'])),
                    bndbox_node)
    createChildNode(doc, 'ymin', str(int(attrs['bndbox']['ymin'])),
                    bndbox_node)
    createChildNode(doc, 'xmax', str(int(attrs['bndbox']['xmax'])),
                    bndbox_node)
    createChildNode(doc, 'ymax', str(int(attrs['bndbox']['ymax'])),
                    bndbox_node)
    object_node.appendChild(bndbox_node)

    return object_node


# 将documentElement写入XML文件
def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')
    doc.writexml(tmpfile, addindent='' * 4, newl='\n', encoding='utf-8')
    tmpfile.close()
    # # 删除第一行默认添加的标记
    fin = open('tmp.xml')
    fout = open(filename, 'w')
    lines = fin.readlines()
    for line in lines[1:]:
        if line.split():
            fout.writelines(line)
    fin.close()
    fout.close()


if __name__ == "__main__":
    ##json文件路径和图片路径,
    json_path = "D:/liyiman/project_znv/mytool/jsonToXml/jsonDataSet/Annotations/"
    img_path = "D:/liyiman/project_znv/mytool/jsonToXml/xmlDataSet/JPEGImages/"
    json_list = os.listdir(json_path)
    # print("json_list:", json_list)
    fileList = os.listdir(img_path)
    # print(".....::")
    # print("fileList:", fileList)
    if fileList == 0:
        os._exit(-1)
        # 对于每一张图都生成对应的json文件
    for imageName in fileList:
        saveName = imageName.replace(".jpg", "")
        print("图片名称:", saveName)
        # 得到xml文件的名字
        xml_file_name = os.path.join(_ANNOTATION_SAVE_PATH, (saveName + '.xml'))
        # print(...)
        # print("xml_file_name:", xml_file_name)
        try:
            img = cv2.imread(os.path.join(img_path, imageName))
            # print(os.path.join(img_path, imageName))
            # cv2.imshow(img)
        except:
            print(imageName)
        height, width, channel = img.shape
        # print(height, width, channel)
        my_dom = xml.dom.getDOMImplementation()
        doc = my_dom.createDocument(None, _ROOT_NODE, None)
        # 获得根节点
        root_node = doc.documentElement
        # folder节点
        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)
        # filename节点
        createChildNode(doc, 'filename', saveName + '.jpg', root_node)
        print("正在创建各个结点中")
        # source节点
        source_node = doc.createElement('source')
        # source的子节点
        createChildNode(doc, 'database', _DATABASE_NAME, source_node)
        # createChildNode(doc, 'annotation', _ANNOTATION, source_node)
        # createChildNode(doc, 'image', 'flickr', source_node)
        root_node.appendChild(source_node)
        size_node = doc.createElement('size')
        createChildNode(doc, 'width', str(width), size_node)
        createChildNode(doc, 'height', str(height), size_node)
        createChildNode(doc, 'depth', str(channel), size_node)
        root_node.appendChild(size_node)
        # 创建segmented节点
        createChildNode(doc, 'segmented', _SEGMENTED, root_node)
        # print("创建object节点")
        ann_data = []
        for i in json_list:
            json_path1 = json_path + i
            with open(json_path1, "r") as f:
                ann = json.load(f)
            imgName = "" + str(ann["file_name"])
            imgName = imgName.replace(".jpg", "")
            # print("次循环内的imgName", imgName)
            cname = saveName
            # print("次循环内的savename:", saveName)
            if (saveName == imgName):
                # object节点
                # print(ann)
                # print("tupianm", ann["imagePath"])
                for j in range(len(ann['detail'])):
                    # print("该图片中有几个框:", len(ann["shapes"]))
                    object_node = createObjectNode(doc, ann["detail"][j])
                    root_node.appendChild(object_node)
            else:
                continue
        # 构建XML文件名称
        print(xml_file_name)  # 写入文件
        writeXMLFile(doc, xml_file_name)