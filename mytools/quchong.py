import hashlib
import os
import cv2
import numpy as np


files_path = "D:/baidu_shinei/"
save_files_path = "D:/baidu_shinei_0/"
files = os.listdir(files_path)  # 遍历文件夹下的所有文件

temp = set()  # 创建一个set()
count = 0  # 删除的文件计数
for file in files:
    file_path = files_path + file  # 获得完整的路径
    # if file.endswith('.jpg'):
    img = cv2.imread(file_path)
    try:
        size = img.shape
    except:
        continue
    # if size[0] > 400 and size[1] > 400:

    img_array = np.array(img)  # 转为数组
    md5 = hashlib.md5()  # 创建一个hash对象
    md5.update(img_array)  # 获得当前文件的md5码
    if md5.hexdigest() not in temp:  # 如果当前的md5码不在集合中
        temp.add(md5.hexdigest())  # 则添加当前md5码到集合中
        cv2.imwrite(save_files_path + md5.hexdigest() + '.jpg', img)
    else:
        count += 1  # 否则删除图片数加一
print(count)
