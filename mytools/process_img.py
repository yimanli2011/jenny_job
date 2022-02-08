#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# import os
# from PIL import Image
#
# #将图像缩小到一定范围
# def size_down(img_file):
#     im = Image.open(img_file)
#     (x, y) = im.size
#     x_s = x
#     y_s = y
#     if x > y and x > 800:
#         x_s = 400
#         y_s = int(y * x_s / x)
#     if x <= y and y > 800:
#         y_s = 400
#         x_s = int(x * y_s / y)
#     out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
#     out.save(img_file)
#
# def list_files(path):
#     files_list = []
#     for root, _, files in os.walk(path):
#         files_list += files
#     return files_list
#
# imgs_root = './qixing_imgs_4_gx/'
# imgs = list_files(imgs_root)
# for file in imgs:
#     size_down(imgs_root + file)


# *******Start*******检测图像数据是否有损坏，避免出现PIL Image "image file is truncated"问题******************
from PIL import Image
import os
import cv2
import shutil
import numpy as np

file_path = 'D:/liyiman/datasets/Face_attr_mask/facemask/JPEGImages/'
to_path = 'D:/liyiman/datasets/Face_attr_mask/facemask/JPEGImages/'
out_file = 'D:/liyiman/datasets/Face_attr_mask/facemask/posun_imgs_done/'
posun_list = []
os.makedirs(to_path, exist_ok=True)
os.makedirs(out_file, exist_ok=True)


def re_write(img_path, out_file):
    # 读入原图片
    img = cv2.imread(img_path)
    if img is None:
        print(img_path)
    else:
        # resize后保存
        # img = cv.resize(img, (720, 512), interpolation=cv.INTER_CUBIC)
        # 重新保存
        img = np.clip(img, 0, 255)
        cv2.imwrite(os.path.join(out_file, os.path.split(img_path)[-1]), img)

# 1、将有问题图片移动到posun文件夹
# for file in os.listdir(file_path):
#     try:
#         src = Image.open(file_path + file).convert('RGB')
#         img = cv2.imread(file_path + file)
#         if img is None:
#             posun_list.append(file)
#     except:
#         posun_list.append(file)
#
#
# for img in posun_list:
#     old_path = os.path.join(file_path, img)
#     new_path = os.path.join(to_path, img)
#     shutil.move(old_path, new_path)
#     img_ = cv2.imread(new_path)
#     if img_ is None:
#         # 图片未能正常读入的原因如果是非正常修改后缀名，
#         # 应用PIL包的Image模块将图片重新转换格式为jpg(彻底转换格式，而不是只改后缀)
#         img = Image.open(new_path).convert('RGB')
#         img.save(new_path.replace('.png', '.jpg'))

# 2、将其中可以正常打开的图像缩重新保存后，就不会出现Corrupt JPEG data: premature end of data segment的报错
img_list = os.listdir(to_path)
for img_file in img_list:
    re_write(os.path.join(to_path, img_file), out_file)

# *************检测图像数据是否有损坏，避免出现PIL Image "image file is truncated"问题****End**********

# import os
# import cv2 as cv
# # 将pix2pix成对的图像拆分成单独的图像文件
# imgs_root = 'D:/liyiman/mygit/pytorch-CycleGAN-and-pix2pix/datasets/datasets/night2day/test/'
# out_file = 'D:/liyiman/mygit/pytorch-CycleGAN-and-pix2pix/datasets/datasets/night2day/test_done/'
# os.makedirs(out_file, exist_ok=True)
# # 初始化新生成的图片名称
# new_img_dict = {}
# for file in os.listdir(imgs_root):
#     # 读入原图片
#     img = cv.imread(os.path.join(imgs_root, file))
#     if img is None:
#         print(file)
#     else:
#         height, width = img.shape[0], img.shape[1]
#         width_new = int(width / 2)
#         # 拆分到2个文件夹，对相同文件名的图片仅生成1次
#         item1 = os.path.splitext(file)[0].split('_')[1]
#         item2 = os.path.splitext(file)[0].split('_')[3]
#         if item1 not in new_img_dict.keys():
#             new_img_dict[item1] = 0
#             img_name_new = item1 + os.path.splitext(file)[1]
#             img_new = img[:, 0:width_new, :]
#             out_dir = out_file + 'night/'
#             os.makedirs(out_dir, exist_ok=True)
#             cv.imwrite(os.path.join(out_dir, img_name_new), img_new)
#         if item2 not in new_img_dict.keys():
#             new_img_dict[item2] = 0
#             img_name_new = item2 + os.path.splitext(file)[1]
#             img_new = img[:, width_new:min(width_new * 2, width), :]
#             out_dir = out_file + 'day/'
#             os.makedirs(out_dir, exist_ok=True)
#             cv.imwrite(os.path.join(out_dir, img_name_new), img_new)
#
#         # 拆分到1个文件夹，对相同文件名的图片，重命名并生成
#         # for i in range(2):
#         #     item = os.path.splitext(file)[0].split('_')[i*2+1]
#         #     if item not in new_img_dict.keys():
#         #         new_img_dict[item] = 0
#         #         img_name_new = item + os.path.splitext(file)[1]
#         #     else:
#         #         # continue
#         #         new_img_dict[item] += 1
#         #         img_name_new = item + '_' + str(new_img_dict[item]) + os.path.splitext(file)[1]
#         #     img_new = img[:, (width_new * i):min(width_new * (i+1), width), :]
#         #     cv.imwrite(os.path.join(out_file, img_name_new), img_new)


