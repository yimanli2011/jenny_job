#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import cv2
import os

video_dir = r'D:\liyiman\datasets\data_chaifen\tebian\shipin'
video_list = [x for x in os.listdir(video_dir) if os.path.splitext(x)[-1] == ".mp4"]
for i in range(len(video_list)):
    # save_file = 'D:/liyiman/datasets/data_chaifen/weigui_shipin_20211109/{}/'.format(os.path.splitext(video_list[i])[0])
    save_file = 'D:/liyiman/datasets/data_chaifen/tebian/{}/'.format('video_'+str(i))
    os.makedirs(save_file, exist_ok=True)
    vc = cv2.VideoCapture(os.path.join(video_dir, video_list[i]))  # 读入视频文件，命名cv
    n = 1  # 计数

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    timeF = 100  # 视频帧计数间隔频率

    j = 0
    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            j += 1
            print(j)
            # cv2.imwrite(save_file + '{}_{}.jpg'.format(os.path.splitext(video_list[i])[0],j*timeF), frame)  # 存储为图像
            cv2.imwrite(save_file + '{}_{}.jpg'.format('video_tebian_'+str(i), j * timeF), frame)  # 存储为图像
        n = n + 1
        cv2.waitKey(1)
    vc.release()