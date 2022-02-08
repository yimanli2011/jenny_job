#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import xml
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm

_IMAGE_PATH = ''

_INDENT = '' * 4
_NEW_LINE = '\n'
_FOLDER_NODE = 'JPEGImages'
_ROOT_NODE = 'annotation'
_SEGMENTED = '0'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'

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
def createObjectNode(doc, name, bbox):
    object_node = doc.createElement('object')

    createChildNode(doc, 'name', name,
                    object_node)
    createChildNode(doc, 'pose',
                    _POSE, object_node)
    createChildNode(doc, 'truncated',
                    _TRUNCATED, object_node)
    createChildNode(doc, 'difficult',
                    _DIFFICULT, object_node)
    bndbox_node = doc.createElement('bndbox')
    # print("midname1[points]:",midname1["points"])
    createChildNode(doc, 'xmin', str(bbox[0]),
                    bndbox_node)
    createChildNode(doc, 'ymin', str(bbox[1]),
                    bndbox_node)
    createChildNode(doc, 'xmax', str(bbox[2]),
                    bndbox_node)
    createChildNode(doc, 'ymax', str(bbox[3]),
                    bndbox_node)
    object_node.appendChild(bndbox_node)

    return object_node


# 将documentElement写入XML文件
def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')
    doc.writexml(tmpfile, addindent='\t', newl='\n', encoding='utf-8')
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


def doc_init(base_name, anno_fn, h, w, c):
    my_dom = xml.dom.getDOMImplementation()
    doc = my_dom.createDocument(None, _ROOT_NODE, None)
    # 获得根节点
    root_node = doc.documentElement
    createChildNode(doc, 'folder', _FOLDER_NODE, root_node)
    createChildNode(doc, 'filename', base_name + '.jpg', root_node)
    createChildNode(doc, 'path', os.path.join(_IMAGE_PATH, base_name + '.jpg'), root_node)

    source_node = doc.createElement('source')
    createChildNode(doc, 'database', 'Unknown', source_node)
    root_node.appendChild(source_node)

    size_node = doc.createElement('size')
    createChildNode(doc, 'width', str(h), size_node)
    createChildNode(doc, 'height', str(w), size_node)
    createChildNode(doc, 'depth', str(c), size_node)
    root_node.appendChild(size_node)

    createChildNode(doc, 'segmented', _SEGMENTED, root_node)
    for item in anno_fn:
        coordinate = [int(i) for i in item.strip().split(' ')]
        l_box = [coordinate[0], coordinate[1], coordinate[0] + coordinate[2], coordinate[1] + coordinate[3]]
        object_node = createObjectNode(doc, 'fire', l_box)
    root_node.appendChild(object_node)

    return doc

# 将视频标签拆为对应图片的标签文件
def video2img_xml(video_xml_file, img_file):
    xml_list = [x for x in os.listdir(video_xml_file) if os.path.splitext(x)[-1] == ".xml"]
    for xml_video in tqdm(xml_list):
        video_name = os.path.splitext(xml_video)[0]
        img_dir = os.path.join(img_file, video_name)
        img_list = [x for x in os.listdir(img_dir) if os.path.splitext(x)[-1]==".jpg"]
        new_xml_file = img_dir + '_label'
        os.makedirs(new_xml_file, exist_ok=True)
        # 获取视频标签文件内容并存入labels字典
        labels = defaultdict(list)
        # 返回解析树
        tree = ET.parse(os.path.join(video_xml_file, xml_video))
        root = tree.getroot()
        frame = root.find('frames')
        for member in frame.findall('_'):
            frameNumber = member.find('frameNumber').text
            try:
                annotation = member.find('annotations').find('_').text
            except:
                annotation = ''
            labels[frameNumber].append(annotation)

        # 获取图片文件在labels字典中对应帧的label内容，并生成每个图片的xml文件。
        for img in img_list:
            base_name = os.path.splitext(img)[0]
            img_fn = base_name.split('_')[-1]
            anno_fn = labels[img_fn]
            if anno_fn != ['']:
                img = cv2.imread(os.path.join(img_dir, img))
                if img is None:
                    continue
                height, width, channel = img.shape

                doc = doc_init(base_name, anno_fn, height, width, channel)
                writeXMLFile(doc, os.path.join(new_xml_file, base_name + '.xml'))


video_xml_file = r'D:/liyiman/datasets/data_chaifen/furg-fire-dataset-master/'
img_file = r'D:/liyiman/datasets/data_chaifen/furg_fire/'
video2img_xml(video_xml_file, img_file)