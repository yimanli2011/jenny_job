import os
import os.path as osp
import cv2
from tqdm import tqdm
from collections import defaultdict
import shapely
import numpy as np
from shapely.geometry import Polygon, MultiPoint  # 多边形
import json


def cal_iou_polygon(box,pg):
    a = [box[0],[box[0][0],box[1][1]], box[1],[box[1][0],box[0][1]]]
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    # print(Polygon(poly1).convex_hull)  # 可以打印看看是不是这样子(0 0, 0 2, 2 2, 2 0, 0 0)

    b = [pg[0], pg[3], pg[2], pg[1]]
    poly2 = Polygon(pg).convex_hull

    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点;(0 0, 0 2, 1 4, 4 4, 4 1, 2 0, 0 0)
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area  # 最小多边形点面积
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0

    return iou


def cal_iou_line(box, pg):
    a = [box[0][0], box[0][1], box[1][0], box[1][1]]
    b = [pg[0][0], pg[0][1], pg[1][0], pg[1][1]]
    if b[0] < a[0] or b[2] > a[2] or b[1] < a[1] or b[3] > a[3]:
        iou = 0
    else:
        iou = 888
    return iou


def cal_in(box, pg):
    a = [box[0][0], box[0][1], box[1][0], box[1][1]]
    b = [pg[0][0], pg[0][1], pg[1][0], pg[1][1]]
    if b[0] < a[0] or b[2] > a[2] or b[1] < a[1] or b[3] > a[3]:
        iou = 0
    else:
        iou = 888
    return iou


def writejsonFile_line(img_name, w, h,label_dict, save_path):
    shapes_list = []
    for item in label_dict:
        shape_type = 'line'
        for i in range(len(label_dict[item])):
            item_dict = {'label':item,
                         'points':label_dict[item][i],
                         'group_id': None,
                         'shape_type': shape_type,
                         'flags': {}
                         }
            shapes_list.append(item_dict)

    json_dict = {'version': '4.5.7',
                 'flags': {},
                 'shapes': shapes_list,
                 'imagePath': img_name,
                 'imageData': None,
                 'imageHeight': h,
                 'imageWidth': w}

    with open(save_path, 'w') as new_js:
        json.dump(json_dict, new_js)


def writejsonFile_polygon(img_name, w, h,label_dict, save_path):
    shapes_list = []
    # if len(label_dict['chepai'])>1: print("len(label_dict['chepai'])>1:"+img_name)
    for item in label_dict:
        shape_type = 'polygon'
        for i in range(len(label_dict[item])):
            if len(label_dict[item][i]) == 2:
                shape_type = 'circle'
            item_dict = {'label':item,
                         'points':label_dict[item][i],
                         'group_id': None,
                         'shape_type': shape_type,
                         'flags': {}
                         }
            shapes_list.append(item_dict)

    json_dict = {'version': '4.5.7',
                 'flags': {},
                 'shapes': shapes_list,
                 'imagePath': img_name,
                 'imageData': None,
                 'imageHeight': h,
                 'imageWidth': w}

    with open(save_path, 'w') as new_js:
        json.dump(json_dict, new_js)


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


img_root = "D:/liyiman/datasets/ZNV_data/bileiqi_20220127/imgs/"
json_root = "D:/liyiman/datasets/ZNV_data/bileiqi_20220127/labels_sbp/"
target_json_root = "D:/liyiman/datasets/ZNV_data/bileiqi_20220127/bileiqi_sbp_cut/Annotations"
target_img_root = "D:/liyiman/datasets/ZNV_data/bileiqi_20220127/bileiqi_sbp_cut/JPEGImages"
cut_key = "biaopan_up"


os.makedirs(target_img_root, exist_ok=True)
os.makedirs(target_json_root, exist_ok=True)
img_list = [x for x in os.listdir(img_root) if osp.splitext(x)[-1]==".jpg"]
json_list = [x for x in os.listdir(json_root) if osp.splitext(x)[-1]==".json"]

for json_file in tqdm(json_list):
    base_name = osp.splitext(json_file)[0]
    # print(base_name+".jpg")
    json_path = osp.join(json_root, json_file)
    img = cv2.imread(osp.join(img_root, base_name+".jpg"))
    if img is None:
        continue
    height, width, channel = img.shape
    if height < 50 or width < 50:
        continue

    flag = False
    labels = defaultdict(list)

    num = 1
    with open(json_path, 'r', encoding='utf-8')as js:
        load_dict = json.loads(js.read())
        for i,item in enumerate(load_dict['shapes']):
            name = item['label']
            points = item['points']
            labels[name].append(points)

    for box in labels[cut_key]:
        if box[0][0] > box[1][0]:
            temp = box[1][0]
            box[1][0] = box[0][0]
            box[0][0] = temp
        if box[0][1] > box[1][1]:
            temp = box[1][1]
            box[1][1] = box[0][1]
            box[0][1] = temp
        bbox = [
            int(max(float(box[0][0]),0)),
            int(max(float(box[0][1]),0)),
            int(max(float(box[1][0]),0)),
            int(max(float(box[1][1]),0))
        ]
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crop_h, crop_w, crop_c = crop_img.shape
        cv2.imwrite(os.path.join(target_img_root, base_name + '_%d.jpg' % num), crop_img)

        new_label_dict = defaultdict(list)
        for item in labels:
            if item != cut_key:
                for lst in labels[item]:
                    # 矩形框
                    #if len(lst) != 4:
                    # 直线
                    # if len(lst) != 2:
                    #     continue
                    # 计算是否在该框中
                    # iou = cal_iou_polygon(box, lst)
                    # iou = cal_iou_line(box, lst)
                    iou = cal_in(box, lst)
                    if iou > 0:
                        l_box = []
                        for i in range(len(lst)):
                            l_box.append([max(0, lst[i][0]-box[0][0]), max(0, lst[i][1]-box[0][1])])
                        # l_box = [
                        #          [max(0, lst[0][0]-box[0][0]), max(0, lst[0][1]-box[0][1])],
                        #          [max(0, lst[1][0]-box[0][0]), max(0, lst[1][1]-box[0][1])],
                        #          ]
                        new_label_dict[item].append(l_box)
        # 多边形
        # writejsonFile_polygon(base_name + '_%d.jpg' % num, crop_w, crop_h, new_label_dict, osp.join(target_json_root, base_name + '_%d.json' % num))
        # 直线
        writejsonFile_line(base_name + '_%d.jpg' % num, crop_w, crop_h, new_label_dict, osp.join(target_json_root, base_name + '_%d.json' % num))
        num += 1

