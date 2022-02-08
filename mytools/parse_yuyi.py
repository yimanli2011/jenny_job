#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import json
import numpy as np
import shutil
from tqdm import tqdm


img_root = r"D:\xunlei\black_biaoji"
label_root = r"D:\xunlei\black_biaoji"
Img_root = r"D:\xunlei\biaoji_2\Images"
Ann_root = r"D:\xunlei\biaoji_2\annotations"

img_list = [x for x in os.listdir(img_root) if (os.path.splitext(x)[-1] == ".jpg" or
            os.path.splitext(x)[-1] == ".jpeg")]
label_list = [x for x in os.listdir(label_root) if os.path.splitext(x)[-1] == ".json"]
print(len(img_list))

os.makedirs(Img_root, exist_ok=True)
os.makedirs(Ann_root, exist_ok=True)

n = 513
for file in label_list:
    base_name = os.path.splitext(file)[0]
    if (base_name+".jpg" in img_list) or (base_name+".jpeg" in img_list):
        if os.path.exists((os.path.join(img_root, base_name+".jpg"))):
            shutil.copyfile(os.path.join(img_root, base_name+".jpg"),
                            os.path.join(Img_root, "%03d.jpg"%n))
        else:
            shutil.copyfile(os.path.join(img_root, base_name + ".jpeg"),
                            os.path.join(Img_root, "%03d.jpg"%n))
        shutil.copyfile(os.path.join(label_root, file), os.path.join(Ann_root, "%03d.json"%n))
        # print(file)
        print(os.path.join(Img_root, base_name + "%03d.jpg"%n))
        n += 1

#  parse yibiao
Img_root = r"D:\data\public\yuyi\images"
Ann_root = r"D:\data\public\yuyi\annotations"
mask_root = r"D:\data\bileiqi_up_cut\bileiqi_up_cut\mask"
os.makedirs(mask_root, exist_ok=True)

label_list = os.listdir(Ann_root)
for file in tqdm(label_list):
    # print(file)
    with open(os.path.join(Ann_root, file), 'r', encoding='utf-8') as json_data:
        data = json.load(json_data)
        appix = os.path.splitext(data['imagePath'])[-1]
        width = data['imageWidth']
        height = data['imageHeight']
        infos = data['shapes']
        mask = np.zeros((int(height), int(width))).astype("uint8")
        img = cv2.imread(os.path.join(Img_root, file.split(".")[0]+".jpg"))
        for info in infos:
            points = info['points']
            pt1 = (int(points[0][0]), int(points[0][1]))
            pt2 = (int(points[1][0]), int(points[1][1]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)
            if info['label'] == 'zhizhen':
                cv2.line(mask, pt1, pt2, 1, 2)
            else:
                cv2.line(mask, pt1, pt2, 2, 2)
        # cv2.imwrite(os.path.join(mask_root, file.split(".")[0]+".png"), mask)
        imgs = np.concatenate((img, np.repeat(mask[...,np.newaxis],3,2)*100), axis=1)

        cv2.imshow("mask", imgs)
        cv2.waitKey(0)

#  transfer yuyi_yibiao
Img_root = r"D:\data\bileiqi_up_cut\bileiqi_up_cut\JPEGImages"
Ann_root = r"D:\data\bileiqi_up_cut\bileiqi_up_cut\mask"
tar_img = r"D:\data\public\yuyi\images"
tar_ann = r"D:\data\public\yuyi\mask"

for i, file in enumerate(os.listdir(Img_root)):
    mask_name = os.path.splitext(file)[0] + ".png"
    shutil.copyfile(os.path.join(Img_root, file), os.path.join(tar_img, str(i+563)+".jpg"))
    shutil.copyfile(os.path.join(Ann_root, mask_name), os.path.join(tar_ann, str(i+563)+".png"))