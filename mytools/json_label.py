#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import json
import glob
import shutil

path = r'D:/liyiman/datasets/ZNV_data/bileiqi_20220127/labels_xbp/'
new_dir = r'D:/liyiman/datasets/ZNV_data/bileiqi_20220127/aa/'
os.makedirs(new_dir, exist_ok=True)
# 统计json文件中label
# from collections import defaultdict
# # object_list = []
# object_dict = defaultdict(int)
# # 车道线检测
# object_list = ['continuous white','dashed white','continuous yellow','dashed yellow',
#                'double continuous white','double continuous yellow','unknown','infer',
#                'continuous dashed white','continuous dashed yellow','dashed continuous white',
#                'dashed continuous yellow','double dashed white','double dashed yellow']
# for kw in object_list:
#     object_dict[kw] = 0
#
# move_list = []
# for json_file in glob.glob(path + '/*.json'):
#     with open(json_file, 'r', encoding='utf-8')as js:
#         load_dict = json.loads(js.read())
#         if load_dict['shapes'] == []:
#             move_list.append(json_file)
#         for i,item in enumerate(load_dict['shapes']):
#             # if len(load_dict['shapes']) > 1:
#             #     print(json_file)
#             # if item['label'] == 'continuous\xa0white':
#             #     move_list.append(json_file)
#             if item['label'] not in object_list:
#                 object_list.append(item['label'])
#                 object_dict[item['label']] = 1
#             else:
#                 object_dict[item['label']] += 1
#
# print("{}个类别及个数：".format(len(object_list)))
# print(object_dict)
# #
# for file in move_list:
#     old_path = file
#     new_path = os.path.join(new_dir, os.path.split(file)[1])
#     shutil.move(old_path, new_path)
# #
# 重命名json中label
new_name = 'biaopan_bileiqi'
for json_file in glob.glob(path + '/*.json'):
    with open(json_file, 'r', encoding='UTF-8')as js:
        load_dict = json.loads(js.read())
        for i,item in enumerate(load_dict['shapes']):
            if item['label'] == 'zhizhen_beileiqi':  # ztc_fugai  ztc_weifugai  ztc_buqueding
                load_dict['shapes'][i]['label'] = new_name
            # 使用新字典替换修改后的字典
        json_dict = load_dict
        # print(json_dict)

        # 将替换后的内容写入原文件
    with open(json_file,'w') as new_js:
        json.dump(json_dict,new_js)


#批量修改json中的内容
# replace_name = 'ztc'
# for json_file in glob.glob(path + '/*.json'):
#     with open(json_file, 'r', encoding='UTF-8')as js:
#         load_dict = json.loads(js.read())
#         old_ = load_dict["imagePath"]
#         new_ = old_.replace(' - 副本', '')
#         load_dict["imagePath"] = new_
#             # 使用新字典替换修改后的字典
#         json_dict = load_dict
#         # print(json_dict)
#
#         # 将替换后的内容写入原文件
#     with open(json_file, 'w') as new_js:
#         json.dump(json_dict,new_js)