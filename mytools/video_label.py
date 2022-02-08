#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import sys
import numpy
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    if isinstance(img, numpy.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    # Read video
    video = cv2.VideoCapture("D:/liyiman/datasets/data_chaifen/furg-fire-dataset-master/barbecue.mp4")
    # 保存视频
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    sz = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = 30
    out = cv2.VideoWriter()
    out.open('output.mp4', fourcc, fps, sz, True)
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    # video.set(cv2.CAP_PROP_POS_FRAMES, 5600)  # 跳帧
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    # 按下了“0”键，视频暂停，开始选择bbox
    key0 = 0  # 标记“Z"是否被按下, 偶数为不追踪， 奇数为追踪
    key2 = False  # 默认警告
    key3 = False
    while True:
        # Read a new frame
        k = cv2.waitKey(1) & 0xff
        ok, frame = video.read()
        if not ok:
            break
        if k == 48:  # 如果按下了“0”键，则重新选择bbox
            key0 += 1
            key2 = False  # 默认警告
            tracker = cv2.TrackerKCF_create()
            bbox = cv2.selectROI(frame, False)
            ok = tracker.init(frame, bbox)
            cv2.destroyAllWindows()
        if k == 49:  # 如果按下了“1”键，则取消bbox
            key0 += 1
        if k == 50:  # 如果按下了“2”键，则改变框的颜色
            key2 = True
        if k == 51:  # 如果按下了“3”键，则显示车牌信息
            key3 = True
        if key0 % 2 == 1:
            ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok and key0 % 2 == 1:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            if key2:
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                # cv2.putText(frame, " ", (int(bbox[0]), int(bbox[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 255, 0), 2)
                frame = cv2ImgAddText(frame, "鲁A 6JIN3", int(bbox[0]), int(bbox[1]) - 25, (0, 0, 255), 22)
            else:
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
                cv2.putText(frame, "Press Line", (int(bbox[0]), int(bbox[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, "KFC Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # Display result
        cv2.imshow("Tracking", frame)
        out.write(frame)
        if k == 27:  # Exit if ESC pressed
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

