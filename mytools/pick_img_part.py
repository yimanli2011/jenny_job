#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import os
import shutil
import random


def pick_img(path_from, path_to):
    pick_num = 50
    list_from = os.listdir(path_from)
    for form_file in list_from:
        list_A = os.listdir(os.path.join(path_from, form_file))
        path_to_f = os.path.join(path_to, form_file)
        os.makedirs(path_to_f, exist_ok=True)
        if len(list_A) > pick_num:
            span = int(len(list_A)/pick_num)
            result_num = [x * span for x in range(pick_num)]
        else:
            result_num = range(len(list_A))
        for i in range(len(list_A)):
            if i in result_num:
                old_path = os.path.join(os.path.join(path_from, form_file), list_A[i])
                new_path = os.path.join(path_to_f, list_A[i])
                shutil.move(old_path, new_path)


def pick_xml(path_from, path_to):
    list_from = os.listdir(path_from)
    os.makedirs(path_to, exist_ok=True)
    for form_file in list_from:
        list_A = os.listdir(os.path.join(path_from, form_file))
        for file in list_A:
            if os.path.splitext(file)[1] == '.xml':
                old_path = os.path.join(os.path.join(path_from, form_file), file)
                new_path = os.path.join(path_to, file)
                shutil.move(old_path, new_path)


def pick_zdy(path_from, path_to):
    list_from = os.listdir(path_from)
    os.makedirs(path_to, exist_ok=True)
    list_A = os.listdir('D:/360MoveData/Users/xuhuan/Desktop/face_det_test/shiwai/Mask')
    for img in list_from:
        if img in list_A:
            old_path = os.path.join(path_from, img)
            new_path = os.path.join(path_to, img)
            shutil.move(old_path, new_path)


# img_path_from 文件夹为双层目录
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path_from", type=str, default=r"D:\liyiman\datasets\data_chaifen\weigui_shanghai_202009")
    parser.add_argument("--img_path_to", type=str, default=r"D:\liyiman\datasets\data_chaifen\rfc_weigui_example")
    args = parser.parse_args()

    pick_img(args.img_path_from, args.img_path_to)
    # pick_xml(args.img_path_from, args.img_path_to)
    # pick_zdy(args.img_path_from, args.img_path_to)