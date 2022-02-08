#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import re
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import shutil
# 获取数据集中各属性的数量，获取各类别的class_weight

# 骑行属性
# attr_num = 16
# class_num = [2, 4, 2, 2, 18, 2, 9, 2, 5, 3, 3, 2, 5, 2, 2, 18]
# class_name = ['性别_女', '性别_男', '年龄0-30', '年龄30-45', '年龄45-60','年龄60+', '未打伞', '打伞', '未戴头盔', '戴头盔',
#               '头盔颜色-黑色', '头盔颜色-白色', '头盔颜色-蓝色', '头盔颜色-粉色', '头盔颜色-橙色', '头盔颜色-红色', '头盔颜色-绿色',
#               '头盔颜色-黄色', '头盔颜色-灰色', '头盔颜色-紫色', '头盔颜色-深灰', '头盔颜色-棕', '头盔颜色-蓝(宝石蓝)',
#               '头盔颜色-蓝(浅灰蓝)', '头盔颜色-蓝(淡蓝)', '头盔颜色-棕(卡其)', '头盔颜色-青',  '头盔颜色-灰(银)',
#               '未戴眼镜', '戴眼镜', '车身颜色-黑', '车身颜色-白', '车身颜色-灰', '车身颜色-黄', '车身颜色-红',
#               '车身颜色-紫', '车身颜色-棕', '车身颜色-绿','车身颜色-蓝',
#               '未戴口罩', '戴口罩', '车型_自行车', '车型_摩托车', '车型_电瓶车', '车型_三轮车', '车型_其他',
#               '未背包', '背包', '背包_其他', '角度_正面', '角度_侧面', '角度_背面', '上衣类型_长袖', '上衣类型_短袖',
#               '上衣款式_净色', '上衣款式_间条', '上衣款式_格子', '上衣款式_图案', '上衣款式_拼接', '无手提包', '有手提包',
#               '无挂牌', '有挂牌', '上衣颜色_黑色', '上衣颜色_白色', '上衣颜色_蓝色', '上衣颜色_粉色', '上衣颜色_橙色',
#               '上衣颜色_红色', '上衣颜色_绿色', '上衣颜色_黄色', '上衣颜色_灰色', '上衣颜色_紫色', '上衣颜色_深灰',
#               '上衣颜色_棕', '上衣颜色_蓝(宝石蓝)', '上衣颜色_蓝(浅灰蓝)', '上衣颜色_蓝(淡蓝)', '上衣颜色_棕(卡其)',
#               '上衣颜色_青', '上衣颜色_灰(银)']
# 驾驶员属性
attr_num = 9
class_num = [2, 2, 2, 2, 2, 2, 2, 2, 2]
class_name = ['是否闭眼_否', '是否闭眼_是', '是否打哈欠_否', '是否打哈欠_是', '是否打电话_否', '是否打电话_是',
              '是否玩手机_否', '是否玩手机_是', '是否抽烟_否', '是否抽烟_是', '是否目视前方_是', '是否目视前方_否',
              '是否系安全带_是', '是否系安全带_否', '是否双手脱离方向盘_否', '是否双手脱离方向盘_是',
              '是否佩戴红外阻挡墨镜_否', '是否佩戴红外阻挡墨镜_是']

# 驾驶员遮挡
# attr_num = 1
# class_num = [2]
# class_name = ['未遮挡', '遮挡']

# 行人属性

# class_name = ['年龄-大于15小于30', '年龄-大于30小于45', '年龄-大于45小于60', '年龄-大于60',
#               '随身携带-背包', '随身携带-其他', '下装款式-休闲装', '上装款式-休闲装', '下装款式-正装', '上装款式-正装',
#               '配饰-帽子', '上装款式-夹克', '下装款式-牛仔', '鞋子款式-皮鞋', '上装Logo-有Logo', '发型-长发',
#               '性别-男', '随身携带-手提袋', '配饰-围巾', '配饰-无', '随身携带-无', '上装图案-格子', '随身携带-塑料袋',
#               '鞋子款式-凉鞋', '鞋子款式-球鞋', '下装款式-短裤', '上装款式-短袖', '下装款式-短裙', '鞋子款式-运动鞋',
#               '上装图案-细条纹', '配饰-太阳镜', '下装款式-长裤', '上装款式-T恤', '上装款式-其他', '上装款式-V领',
#               '上装颜色-黑色', '上装颜色-蓝色', '上装颜色-棕色', '上装颜色-绿色', '上装颜色-灰色', '上装颜色-橙色',
#               '上装颜色-粉色', '上装颜色-紫色', '上装颜色-红色', '上装颜色-白色', '上装颜色-黄色', '下装颜色-黑色',
#               '下装颜色-蓝色', '下装颜色-棕色', '下装颜色-绿色', '下装颜色-灰色', '下装颜色-橙色', '下装颜色-粉色',
#               '下装颜色-紫色', '下装颜色-红色', '下装颜色-白色', '下装颜色-黄色', '头发颜色-黑色', '头发颜色-蓝色',
#               '头发颜色-棕色', '头发颜色-绿色', '头发颜色-灰色', '头发颜色-橙色', '头发颜色-粉色', '头发颜色-紫色',
#               '头发颜色-红色', '头发颜色-白色', '头发颜色-黄色', '鞋子颜色-黑色', '鞋子颜色-蓝色', '鞋子颜色-棕色',
#               '鞋子颜色-绿色', '鞋子颜色-灰色', '鞋子颜色-橙色', '鞋子颜色-粉色', '鞋子颜色-紫色', '鞋子颜色-红色',
#               '鞋子颜色-白色', '鞋子颜色-黄色', '配饰-耳机', '年龄-小于15', '随身携带-婴儿推车', '发型-光头',
#               '鞋子款式-靴子', '下装款式-七分裤', '随身携带-购物车', '随身携带-雨伞', '性别-女', '随身携带-文件夹',
#               '配饰-发带', '下装款式-热裤', '随身携带-头巾', '下装款式-长裙', '上装款式-长袖', '下装图案-格子',
#               '下装图案-细条纹', '随身携带-行李箱', '上装款式-无袖', '发型-短发', '鞋子款式-长筒袜', '上装款式-西装',
#               '随身携带-手提箱', '下装款式-西裤', '上装款式-线衫', '上装图案-宽条纹', '人体姿态-站', '人体姿态-坐',
#               '人体姿态-趴', '人体姿态-躺', '人体姿态-蹲', '人体角度-正面', '人体角度-侧面', '人体角度-背面', '配饰-口罩',
#               '上装图案-净色', '上装图案-图案', '上装图案-拼接', '下装图案-净色', '下装图案-图案', '上装颜色-深灰',
#               '上装颜色-蓝(宝石蓝)', '上装颜色-蓝(浅灰蓝)', '上装颜色-蓝(淡蓝)', '上装颜色-棕(卡其)', '上装颜色-青色',
#               '上装颜色-灰(银)', '下装颜色-深灰', '下装颜色-蓝(宝石蓝)', '下装颜色-蓝(浅灰蓝)', '下装颜色-蓝(淡蓝)',
#               '下装颜色-棕(卡其)', '下装颜色-青色', '下装颜色-灰(银)']

# 人脸属性
# attr_num = 10
# class_num = [2, 3, 2, 2, 2, 2, 2, 3, 2, 5]
# class_name = ['male', 'female',
#               'no_eyeglass', 'eyeglass', 'sunglass',
#               'eyeclose', 'eyeopen',
#               'no_smile', 'smile',
#               'no_mask', 'mask',
#               'mouthclose', 'mouthopen',
#               'no_beard', 'beard',
#               'yellow', 'black', 'white',
#               'no_cap', 'cap',
#               'happy', 'calm', 'angry', 'sorrow', 'surprised']
# CLASSES = [['age_lower_' + str(i) for i in range(101)],
#            ['age_up_' + str(i) for i in range(101)],
#            ['age_' + str(i) for i in range(101)],
#            ['attract_' + str(i) for i in range(101)],
#            ['male', 'female'],
#            ['no_eyeglass', 'eyeglass', 'sunglass'],
#            ['eyeclose', 'eyeopen'],
#            ['no_smile', 'smile'],
#            ['no_mask', 'mask'],
#            ['mouthclose', 'mouthopen'],
#            ['no_beard', 'beard'],
#            ['yellow', 'black', 'white'],
#            ['no_cap', 'cap'],
#            ['happy', 'calm', 'angry', 'sorrow', 'surprised']]

# 文件路径
label_path = [
              # "D:/liyiman/datasets/Face_attr_mask/test_20211122_new.txt"
              "D:/liyiman/datasets/ZNV_data/driver_attr_01/train_1104_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_02/trainval_1104_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_03/train_1104_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_04/trainval_1104_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_05/train_1104_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_06/trainval_1104_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_07/trainval_1104_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_08/trainval_1104_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_1028/trainval_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_20211104/trainval_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_20211201/trainval_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_20211208/trainval_new.txt",
              "D:/liyiman/datasets/ZNV_data/driver_attr_20211216/trainval_new.txt",
              ]
class_count = sum(class_num)
print('class_count:'+ str(class_count))


def one_hot(lst,img):
    lst_onehot = []
    for i in range(len(class_num)):
        attr_one = [0] * class_num[i]
        # if isinstance(lst[i], str):
        #     print(img)
        id = int(lst[i])
        if id < 0:
            attr_one = [id]*len(attr_one)
        else:
            attr_one[id] = 1
        lst_onehot.append(attr_one)
    return lst_onehot


def txt2hot():
    onehot_label = {}
    for i in range(len(label_path)):
        with open(label_path[i],'r',encoding="utf-8") as f:
            # 定义一个用于切割字符串的正则
            seq = re.compile(" ")
            for line in f:
                img_label = []
                lst = seq.split(line.strip())
                img_name = lst[0]
                if img_name in onehot_label.keys():
                    print(label_path[i].split('/')[-2] + '/' + img_name)
                # 人脸属性改为5:，其他为1:
                label_onehot = one_hot(lst[1:], img_name)
                for attr in label_onehot:
                    for item in attr:
                        img_label.append(item)
                onehot_label[img_name] = img_label
                # if img_label[-1] == 1:
                #     print(img_name)
    return onehot_label


def compute_weights(label_dict):
    count_dict = {}
    valid_cnt = torch.tensor([0] * class_count)
    value = torch.tensor(list(label_dict.values()))
    valid_cnt += (value != -1).sum(dim=0)
    value[value == -1] = 0
    pos_ratio = value.sum(dim=0).float() / valid_cnt.float()
    print((1 - pos_ratio).float().exp())
    print(valid_cnt)
    count_list = value.sum(dim=0).tolist()
    for i in range(len(count_list)):
        count_dict[class_name[i]] = count_list[i]
    print(count_dict)


def count_label(label_path):
    count_dict = {}
    class_dict = {}
    for i in range(len(label_path)):
        with open(label_path[i],'r',encoding="utf-8") as f:
            # 定义一个用于切割字符串的正则
            seq = re.compile(" ")
            for line in f:
                lst = seq.split(line.strip())
                img_name = lst[0]
                img_label = [int(i) for i in lst[1:]]
                count_dict.update({img_name:img_label})

    value = torch.tensor(list(count_dict.values()))
    value[value == -1] = 0
    count_list = value.sum(dim=0).tolist()
    for i in range(len(count_list)):
        class_dict[class_name[i]] = count_list[i]
    print(class_dict)

def stat_size(label_path):
    # sizeList = []
    from collections import defaultdict
    size_dict = defaultdict()
    size_ratio_dict = {}
    pixelList = []
    heightList = []
    widthList = []
    num_img = 0
    for i in range(len(label_path)):
        with open(label_path[i], 'r', encoding="utf-8") as f:
            # 定义一个用于切割字符串的正则
            seq = re.compile(" ")
            for line in f:
                lst = seq.split(line.strip())
                img_name = lst[0]
                try:
                    img_path = os.path.join(os.path.split(label_path[i])[0], img_name)
                    img = cv2.imread(img_path)
                    height, width = img.shape[0], img.shape[1]
                    size = (height, width)
                    if size not in size_dict.keys():
                        size_dict[size] = 1
                    else:
                        size_dict[size] += 1
                    # pixel = np.sqrt(height * width)
                    # size = os.path.getsize(img_path)
                    # sizeList.append(size)
                    # heightList.append(height)
                    # widthList.append(width)
                    # pixelList.append(pixel)
                    num_img += 1
                except:
                    print('异常图片：{}'.format(img_path))
                    continue
    # print(pd.Series(pixelList).describe())
    # print('height:')
    # print(pd.Series(heightList).describe())
    # print('width:')
    # print(pd.Series(widthList).describe())

    for keys, item in size_dict.items():
        size_ratio_dict[keys] = format((item/num_img * 100), '.2f') + '%'
    print(size_dict)
    print(size_ratio_dict)
    # fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 6))
    # # 第二个参数是柱子宽一些还是窄一些，越大越窄越密,数据多的时候将这个数值写大一些。
    # ax0.hist(pixelList, 10, density=1, histtype='bar', facecolor='yellowgreen', alpha=0.75, stacked=True)
    # ##pdf概率分布图，一万个数落在某个区间内的数有多少个
    # ax0.set_title('pdf')
    # ax1.hist(pixelList, 10, density=1, histtype='bar', facecolor='pink', alpha=0.75, cumulative=True, rwidth=0.8, stacked=True)
    # # cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
    # ax1.set_title("cdf")
    # fig.subplots_adjust(hspace=0.4)
    # plt.show()
    return pixelList


def pick_img_as_attr(label_path):
    file_path = './mask_test_1_list.txt'
    new_file = './copy_mask_test_1_imgs/'
    match_imgs = []
    for i in range(len(label_path)):
        with open(label_path[i],'r',encoding="utf-8") as f:
            # 定义一个用于切割字符串的正则
            seq = re.compile(" ")
            for line in f:
                lst = seq.split(line.strip())
                img_name = lst[0]
                img_label = [int(i) for i in lst[1:]]
                if img_label[-6] == 1:  # 根据需要修改
                    match_imgs.append(img_name)
    with open(file_path, 'w') as f:
        f.write('\n'.join(match_imgs))
    for i in range(len(match_imgs)):
        old_path = os.path.join(os.getcwd(), match_imgs[i])
        new_path = os.path.join(new_file, match_imgs[i])
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)
        shutil.copy(old_path, new_path)

# import  pickle
# # 重点是rb和r的区别，rb是打开二进制文件，r是打开文本文件
# f=open('D:/360MoveData/Users/xuhuan/Desktop/out.pkl','rb')
# data = pickle.load(f)
# print(data)


# 骑行属性、驾驶员属性、人脸属性类别及数量统计、class_weight权重计算
compute_weights(txt2hot())

# 统计图片大小
# stat_size(label_path)

# 行人属性类别及数量统计
# count_label(label_path)

# 根据属性值查找相应图片
# pick_img_as_attr(label_path)

