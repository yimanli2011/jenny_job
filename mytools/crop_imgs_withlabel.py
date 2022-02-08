import os
import os.path as osp
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from xml_creator import *
import xml
from xml.dom import minidom


_IMAGE_PATH = ' '

_INDENT = '' * 4
_NEW_LINE = '\n'
_FOLDER_NODE = 'JPEGImages'
_ROOT_NODE = 'annotation'
_SEGMENTED = '0'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'


def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou

img_root = "D:/liyiman/datasets/ZNV_data/rfc_SFWear_Det_shida03/VOC2007/JPEGImages/"
xml_root = "D:/liyiman/datasets/ZNV_data/rfc_SFWear_Det_shida03/VOC2007/Annotations/"
target_xml_root = "D:/liyiman/datasets/ZNV_data/rfc_SFWear_Det_shida03/VOC2007/Annotations_new"
target_img_root = "D:/liyiman/datasets/ZNV_data/rfc_SFWear_Det_shida03/VOC2007/JPEGImages_new"

os.makedirs(target_img_root, exist_ok=True)
os.makedirs(target_xml_root, exist_ok=True)
img_list = [x for x in os.listdir(img_root) if osp.splitext(x)[-1]==".jpg"]
xml_list = [x for x in os.listdir(xml_root) if osp.splitext(x)[-1]==".xml"]

for xml_file in tqdm(xml_list):
    base_name = osp.splitext(xml_file)[0]
    print(base_name+".jpg")
    xml_path = osp.join(xml_root, xml_file)
    img = cv2.imread(osp.join(img_root, base_name+".jpg"))
    if img is None:
        continue
    height, width, channel = img.shape
    if height < 50 or width < 50:
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()
    flag = False
    labels = defaultdict(list)

    num = 1
    for obj in root.findall('object'):
        name = obj.find('name').text
        bnd_box = obj.find('bndbox')
        # Coordinates may be float type
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]
        labels[name].append(bbox)
    for box in labels["person"]:
        crop_img = img[box[1]:box[3], box[0]:box[2]]
        crop_h, crop_w, crop_c = crop_img.shape
        cv2.imwrite(os.path.join(target_img_root, base_name + '_%d.jpg' % num), crop_img)

        my_dom = xml.dom.getDOMImplementation()
        doc = my_dom.createDocument(None, _ROOT_NODE, None)
        # 获得根节点
        root_node = doc.documentElement
        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)
        createChildNode(doc, 'filename', base_name + '_%d.jpg'%num, root_node)
        createChildNode(doc, 'path', osp.join(_IMAGE_PATH, base_name + '_%d.jpg' % num), root_node)

        source_node = doc.createElement('source')
        createChildNode(doc, 'database', 'Unknown', source_node)
        root_node.appendChild(source_node)

        size_node = doc.createElement('size')
        createChildNode(doc, 'width', str(crop_w), size_node)
        createChildNode(doc, 'height', str(crop_h), size_node)
        createChildNode(doc, 'depth', str(crop_c), size_node)
        root_node.appendChild(size_node)

        createChildNode(doc, 'segmented', _SEGMENTED, root_node)

        # object_node = createObjectNode(doc, "person", box)
        # root_node.appendChild(object_node)

        for item in labels:
            if item != "person" and item != "truck" and item != "car" and item != "tricycle":
                for lst in labels[item]:
                    iou = cal_iou(box, lst)
                    if iou > 0:
                        l_box = [max(0, lst[0]-box[0]), max(0, lst[1]-box[1]), min(lst[2]-box[0], crop_w),
                                 min(lst[3]-box[1], crop_h)]
                        object_node = createObjectNode(doc, item, l_box)
                        root_node.appendChild(object_node)
        writeXMLFile(doc, osp.join(target_xml_root, base_name + '_%d.xml' % num))
        num += 1

