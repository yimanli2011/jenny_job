#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import os, sys


def max_min_change(file_path):  # 用于将json中points的大小颠倒的进行修正
    change = False
    with open(file_path, 'rb') as f:
        fileJson = json.load(f)
        shapes = fileJson["shapes"]
        for i in range(len(shapes)):
            points = shapes[i]["points"]
            if points[0][0]>points[1][0] or points[0][1]>points[1][1]:
                temp = points[0]
                points[0] = points[1]
                points[1] = temp
                change = True
    f.close()
    return fileJson,change


def rewrite_json_file(filepath, json_data):
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=4)
    f.close()


if __name__ == '__main__':
    # 先修改path文件夹路径
    path = 'test_json'
    file_list = os.listdir(path)
    count = 0
    for i in file_list:
        file_path = path + '/' + i
        m_json_data, change = max_min_change(file_path)
        if change:
            count=count + 1
            print("修改文件为："+i)
        rewrite_json_file(file_path, m_json_data)
    print("修改了"+ str(count) +"个文件")