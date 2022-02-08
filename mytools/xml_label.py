#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os.path
import glob
import xml.etree.ElementTree as ET
import shutil
import numpy as np


# pathlist = [
#             r'D:\liyiman\datasets\ZNV_data\JDWG_Det_shinei\VOC2007\weigui_huangpu_20220113-lable',
#            ]
# object_list = []

# 街道违规
pathlist = [r'D:/liyiman/datasets/ZNV_data/JDWG_Det_shinei/VOC2007/Annotations',
            # r'D:\liyiman\datasets\ZNV_data\JDWG_Det_shinei\VOC2007\weigui_huangpu_20220113-lable'
            # r'D:/liyiman/datasets/ZNV_data/JDWG_Det_02/VOC2007/Annotations/',
            # r'D:/liyiman/datasets/ZNV_data/JDWG_Det_03/VOC2007/Annotations/',
            # r'D:/liyiman/datasets/ZNV_data/JDWG_Det_04/VOC2007/Annotations/',
            # r'D:/liyiman/datasets/ZNV_data/JDWG_Det_05/VOC2007/Annotations/',
            # r'D:/liyiman/datasets/ZNV_data/JDWG_Det_06/VOC2007/Annotations/',
            # r'D:/liyiman/datasets/ZNV_data/JDWG_Det_07/VOC2007/Annotations/',
            ]
# #
# object_list = ['liudongxiaofan', 'ditanxiaofan', 'tanweixiaofan', 'feipinshougou', 'kuamen', 'dianwai', 'yaoyaoche', 'zhandao', 'dabaolaji', 'baolulaji', 'zhixianglaji',
#         'yehuaqiguan', 'bandengjiaju', 'xiaotuiche', 'saozhoutuoba', 'jianzhulaji', 'waizhishuicao', 'chengsandizuo', 'patijiazi', 'lajimanyi',
#         'lajixiangweigai', 'lajixiang', 'lajitong', 'qitalajitong', 'lajituiche', 'lajixiangdaodi', 'menceguanggao', 'guanggaoban', 'dengxiangguanggao',
#         'caihongmen', 'qizhiguanggao', 'hengfu', 'tiaofuguanggao', 'lupaiguanggao', 'zhantieguanggao', 'guanggaoposun', 'zhechuangguanggao', 'chengsan',
#         'zhangpeng', 'liangshai', 'kongtiao', 'jiepo', 'jishui', 'shigong', 'lumianposun', 'jinggaiposun', 'fangzhui', 'fangzhuangzhu', 'shuima', 'hulan',
#         'shebeizawu', 'shenghuozawu']

# 室内违规
object_list = ['dabaolaji', 'baolulaji', 'lajimanyi',
        'lajixiangweigai', 'lajixiang', 'lajitong',
        'qitalajitong', 'lajituiche', 'lajixiangdaodi']




# 抽烟打电话玩手机
# pathlist = [r'D:/liyiman/datasets/ZNV_data/phone_smoke_Det01_c/VOC2007/Annotations/',
#             r'D:/liyiman/datasets/ZNV_data/phone_smoke_Det02/VOC2007/Annotations/']

# object_list = ['playing', 'smoking', 'calling']

# 工程车识别
# pathlist = [r'D:/liyiman/datasets/ZNV_data/CHACHE/VOC2007/Annotations/',]

# object_list = ['chache', 'dumper', 'ffcar']

# 安全穿戴
# pathlist = [r'D:/liyiman/datasets/ZNV_data/SFWear_Det_c/VOC2007/Annotations/',
#             r'D:/liyiman/datasets/ZNV_data/SFWear_Det_shida_c/VOC2007/Annotations/',
#             r'D:/liyiman/datasets/ZNV_data/SFWear_Det_shida02_c/VOC2007/Annotations/',
#             r'D:/liyiman/datasets/ZNV_data/SFWear_Det_shida03_c/VOC2007/Annotations/']

# object_list = ['head', 'hat', 'smoking', 'vest', 'calling', 'playing',
#                'hand', 'glove', 'sleeve', 'oversleeve', 'shoes', 'boots',
#                'nosafeglasses', 'safeglasses']

# 人非车
# pathlist = [
#             # r'D:/liyiman/datasets/ZNV_data/rfc_chepai_qingdao/VOC2007/Annotations/',
#             # r'D:/liyiman/datasets/ZNV_data/rfc_chepai_shanghai/VOC2007/Annotations/',
#             # r'D:/liyiman/datasets/ZNV_data/rfc_Det_01/VOC2007/Annotations/',
#             # r'D:/liyiman/datasets/ZNV_data/rfc_Det_02/VOC2007/Annotations/',
#             # r'D:/liyiman/datasets/ZNV_data/rfc_Det_03/VOC2007/Annotations/',
#             # r'D:/liyiman/datasets/ZNV_data/rfc_Det_04/VOC2007/Annotations/',
#             r'D:/liyiman/datasets/ZNV_data/rfc_Det_05/VOC2007/Annotations/'
#            ]
#
# object_list = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'rider', 'tricycle', 'chepai']

# 船舶
# pathlist = [r'D:/liyiman/datasets/ZNV_data/ship_Det_01/VOC2007/Annotations/']
#
# object_list = ['ore_carrier', 'cargo_ship', 'container_ship', 'fishing_boat', 'passenger_ship',
#                'speed_boat', 'oar_boat', 'sail_boat', 'excursion_boat','unknown']

# 重命名xml中label
# for path in pathlist:
#     for xml_file in glob.glob(path + '/*.xml'):
#     ####### 返回解析树
#         tree = ET.parse(xml_file)
#         ##########获取根节点
#         root = tree.getroot()
#         #######对所有目标进行解析
#         for member in root.findall('object'):
#             objectname = member.find('name').text
#             if objectname == 'lajitongd':
#                 # print(xml_file)
#                 member.find('name').text = str('lajixiangdaodi')
#                 tree.write(xml_file)

# def get_median(data):
#     data.sort()
#     half = len(data) // 2
#     return (data[half] + data[~half]) / 2
# def get_average(list):
#     sum = 0
#     for item in list:
#         sum += item
#     return sum/len(list)

# 查看xml文件中各label数量
from collections import defaultdict
from tqdm import tqdm
object_dict = defaultdict(int)
# # object_size = defaultdict()
# # object_scale = defaultdict()
#
for kw in object_list:
    object_dict[kw] = 0
#     # object_size[kw] = []
#     # object_scale[kw] = []
img_count = 0
null_file = []
find_file = []
for path in pathlist:
    for xml_file in tqdm(glob.glob(path + '/*.xml')):
        img_count += 1
        ####### 返回解析树
        tree = ET.parse(xml_file)
        ##########获取根节点
        root = tree.getroot()

        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        #######对所有目标进行解析
        if len(root.findall('object')) == 0:
            null_file.append(xml_file)
        for member in root.findall('object'):
            objectname = member.find('name').text
            # bnd_box = member.find('bndbox')
            # bbox = [
            #     int(float(bnd_box.find('xmin').text)),
            #     int(float(bnd_box.find('ymin').text)),
            #     int(float(bnd_box.find('xmax').text)),
            #     int(float(bnd_box.find('ymax').text))
            # ]
            # # 计算框的大小
            # ob_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # # 计算框占图片的比例
            # try:
            #     ob_scale = float(format(ob_size/(width * height)*100, '.2f'))
            # except:
            #     continue
            # if objectname == "t":
            #     find_file.append(xml_file)
            # if objectname == "]":
            #     print(xml_file)
            if objectname not in object_list:
                object_list.append(objectname)
                object_dict[objectname] = 1
                # continue
            else:
                object_dict[objectname] += 1
                # find_file.append(xml_file)
            # object_size[objectname].append(ob_size)
            # object_scale[objectname].append(ob_scale)

# # for k, v in object_size.items():
# #     # print("{}的框平均分辨率为{}".format(k, np.sqrt(get_average(v))))
# #     print("{}的框分辨率中位数为{}".format(k, np.sqrt(get_median(v))))
# #     # print("{}的框最大分辨率为{}".format(k, np.sqrt(max(v))))
# #     # print("{}的框最小分辨率为{}".format(k, np.sqrt(min(v))))
# # for k,v in object_scale.items():
# #     # print("{}的框在图中占比平均值为{}".format(k, get_average(v)))
# #     print("{}的框在图中占比中位数为{}".format(k, get_median(v)))
# #     # print("{}的框在图中占比最大值为{}".format(k, max(v)))
# #     # print("{}的框在图中占比最小值为{}".format(k, min(v)))
#
print("共{}张图片".format(img_count))
print("{}个类别及个数：".format(len(object_list)))
print(object_dict)
# print(null_file)

# # 将空标注文件移出
for item in null_file:
    old_path = item
    new_path = item.replace('/Annotations', '/aa')
    os.makedirs(os.path.split(new_path)[0], exist_ok=True)
    shutil.move(old_path, new_path)
#
# 将含有某类别的标注文件移出
# find_file = set(find_file)
# print('移动文件数量为{}'.format(len(find_file)))
# for item in find_file:
#     old_path = item
#     new_path = item.replace('/Annotations', '/aa')
#     os.makedirs(os.path.split(new_path)[0], exist_ok=True)
#     shutil.move(old_path, new_path)



# # 更改xml文件-copy内容加入xml
# import os
# import os.path as osp
# from tqdm import tqdm
#
# copy_from = 'D:/liyiman/datasets/ZNV_data/JDWG_Det_05/VOC2007/zlj-362-20210827/aa.xml'
# xml_root = "D:/liyiman/datasets/ZNV_data/JDWG_Det_05/VOC2007/zlj-362-20210827/label/"
# new_xml_root = 'D:/liyiman/datasets/ZNV_data/JDWG_Det_05/VOC2007/zlj-362-20210827/label_new'
# os.makedirs(xml_root, exist_ok=True)
# os.makedirs(new_xml_root, exist_ok=True)
#
# # 获取要copy的内容
# # 按行读取，按行加入（与一般文档处理方式相同）
# file = open(copy_from, "r")
# object_list = []
# for object in file:
#     object_list.append(object)
# file.close()
#
# xml_list = [x for x in os.listdir(xml_root) if osp.splitext(x)[-1] == ".xml"]
# for xml_file in tqdm(xml_list):
#     base_name = osp.splitext(xml_file)[0]
#     print(base_name + ".xml")
#     old_path = osp.join(xml_root, base_name + ".xml")
#     new_path = osp.join(new_xml_root, xml_file)
#     old_lines = []
#     old_file = open(old_path, "r")
#     for line in old_file:
#         old_lines.append(line)
#     old_file.close()
#     old_lines.pop(-1)
#     for i in range(13, len(object_list)):
#         old_lines.append(object_list[i])
#     output_file = open(new_path, "w")
#     for line in old_lines:
#         output_file.write(line)
#     output_file.close()
#     print("Done!")
#


# 提取xml相关内容保存至txt文件（人脸wiki数据集的处理）
# pathlist = [
#             'D:/liyiman/datasets/Face_attr_mask/face_wiki_images/labels_xml',
#            ]
# txt_path = 'D:/liyiman/datasets/Face_attr_mask/face_wiki_images/wiki_trainval.txt'
# img_count = 0
# null_file = []
# result = []
# for path in pathlist:
#     for xml_file in glob.glob(path + '/*.xml'):
#         img_name = os.path.split(xml_file)[1].replace('.xml', '.jpg')
#         img_count += 1
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         if len(root.findall('object')) == 0:
#             null_file.append(xml_file)
#         if len(root.findall('object')) == 1:
#             member = root.find('object')
#             attr = member.find('props')
#             gender = attr.find('性别').text
#             if gender == '男':
#                 gender = '0'
#             elif gender == '女':
#                 gender = '1'
#             else:
#                 gender = '-1'
#             age = attr.find('年龄').text
#             try:
#                 age = int(age)
#                 age_lower = str(max(0, age - 5))
#                 age_up = str(min(100, age + 5))
#                 age = str(age)
#             except:
#                 age = '-1'
#             attr_str = img_name + ' ' + age_lower + ' ' + age_up + ' ' + age + ' -1 ' + gender \
#                        + ' -1 -1 -1 -1 -1 -1 -1 -1 -1'
#             result.append(attr_str)
#         else:
#             print('异常文件：{}'.format(xml_file))
# with open(txt_path, 'w') as f:
#     f.write('\n'.join(result))
# print("共{}张图片".format(img_count))
# print("以下图片没有标注内容：")
# print(null_file)
