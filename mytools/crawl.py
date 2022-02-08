#!usr/bin/env python
# coding=utf-8

#-*- coding: UTF-8 -*-
import sys
import importlib
importlib.reload(sys)
import os
import os.path
import time
import csv
import codecs
# import pymysql
import pickle
import requests
import re
import urllib.parse
import cv2
import shutil
import random
# import xmltodict
import json
import uuid
from PIL import Image
import xml.dom.minidom
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
######  百度图片爬虫 ######
# 获取动态页面返回的文本
def get_page_html(page_url):
    # headers = {
    #     'Referer': 'https://image.baidu.com/search/index?tn=baiduimage',
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    # }
    # headers = {
    # 'Host': 'image.baidu.com',
    # 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36',
    # 'Cookie': 'BAIDUID=0A4C55C2C07A4AFEA9A2962A4A32E9B6:FG=1; BIDUPSID=0A4C55C2C07A4AFEA9A2962A4A32E9B6; PSTM=1618885599; BDSFRCVID_BFESS=Mh-OJeC62lT9mbceq4wubZYHBetv3QnTH6ao3TWsR1SrLTPC1ZFtEG0P8U8g0Kubmgz-ogKKy2OTH9DF_2uxOjjg8UtVJeC6EG0Ptf8g0f5; H_BDCLCKID_SF_BFESS=tbCO_C0KJCI3qn7I55-_-tAObfo-etJyaR3uBT6vWJ5TMCo135rV-4t8KHOf-4jzymbG04bKtMj_ShPC-tP53h-y0GQgbf7nLgADhqQ23l02VKnae-t2ynLVhfJEe4RMW238oq7mWITUsxA45J7cM4IseboJLfT-0bc4KKJxbnLWeIJIjj6jK4JKjHutqTOP; H_PS_PSSID=33984_33820_31254_33848_33676_33607_34026; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; delPer=0; BAIDUID_BFESS=0A4C55C2C07A4AFEA9A2962A4A32E9B6:FG=1; PSINO=7; __yjs_duid=1_1c2df3da73c373af270afadce2b5f00e1621472126118; ab_sr=1.0.0_ZWEyNTQ3ODMwMWUyZmNmNTM5NWQxNDRlMDkyMWQyMGJhNjI3ZGZiZDkzNmEzM2MyOGUyNGFlOTE0YWEzYTliZjVlZTM1ZmJmMzhkOWQ4NDcwYTBlNjJmYjM3NDAxMTZl; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; BA_HECTOR=0l818h0101040gal0p1gabgv20r'
    # }
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    # }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        'Cookie': 'BAIDUID=0A4C55C2C07A4AFEA9A2962A4A32E9B6:FG=1; BIDUPSID=0A4C55C2C07A4AFEA9A2962A4A32E9B6; PSTM=1618885599; BAIDUID_BFESS=0A4C55C2C07A4AFEA9A2962A4A32E9B6:FG=1; __yjs_duid=1_1c2df3da73c373af270afadce2b5f00e1621472126118; antispam_key_id=23; BDRCVFR[X_XKQks0S63]=mk3SLVN4HKm; userFrom=null; H_PS_PSSID=33984_31254_33848_33676_33607_34026; delPer=0; PSINO=7; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; antispam_data=dcf276ffec46d69c538cc164f729ea9fe81aa618ef99ab013221977406af2be9819b644f71d075093795abf5822af0232d3401195977b3bc5c496c0911c58f18ed13c7ec140ea2786edca0807d119815',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'referer': 'https://graph.baidu.com/s?sign=12151278d51da7fc856b101621924829&f=all&tn=pc&tn=pc&idctag=tc&idctag=tc&sids=10007_10407_10801_10607_10500_10702_10919_10911_11006_10924_10904_10018_10901_10942_10907_11012_10971_10968_10974_11031_12201_17851_17070_18013_18101_19107_18242_19201_17201_17203_18300_18401_18310_18315_18331_18412_19115_19123_19130_19147_9999&sids=10007_10407_10801_10607_10500_10702_10919_10911_11006_10924_10904_10018_10901_10942_10907_11012_10971_10968_10974_11031_12201_17851_17070_18013_18101_19107_18242_19201_17201_17203_18300_18401_18310_18315_18331_18412_19115_19123_19130_19147_9999&logid=3780406733&logid=3780406733&pageFrom=graph_upload_bdbox&pageFrom=graph_upload_pcshitu&srcp=&extUiData%5BisLogoShow%5D=1&tpl_from=pc&entrance=general',
        'content-type': 'application/json; charset=UTF-8',
    }
    try:
        s = requests.session()
        r = s.get(page_url, headers=headers)
        if r.status_code == 200:
            print(r.content.decode('utf-8'))
            r.encoding = r.apparent_encoding
            return r.text
        else:
            print('请求失败')
    except Exception as e:
        print(e)


# 从文本中提取出真实图片地址
def parse_result(text):
    # url_real = re.findall('"thumbURL":"(.*?)",', text)
    url_real = re.findall('"thumbUrl":"(.*?)",', text)
    return url_real


# 获取图片的content
def get_image_content(url_real):
    # headers = {
    #     'Referer': url_real,
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    # }
    headers = {
        # 'Referer': url_real,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    }
    try:
        r = requests.get(url_real, headers=headers)
        if r.status_code == 200:
            r.encoding = r.apparent_encoding
            return r.content
        else:
            print('请求失败')
    except Exception as e:
        print(e)


# 将图片的content写入文件
def save_pic(url_real, content):
    root = 'D://baiduimage//'
    path = root + url_real.split('/')[-1]
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(content)
            print('图片{}保存成功，地址在{}'.format(url_real, path))
    else:
        pass


# 主函数
def main():
    print("="*50)
    # keyword = input('请输入你要查询的关键字: ')
    # keyword_quote = urllib.parse.quote(keyword)
    depth = int(input("请输入要爬取的页数(每页30张图): "))
    for i in range(depth):
        # url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord+=&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&word={}&z=&ic=0&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&step_word={}&pn={}&rn=30&gsm=1e&1541136876386='.format(
        #     keyword_quote, keyword_quote, i * 30)
        url = 'https://graph.baidu.com/ajax/pcsimi?sign=12151278d51da7fc856b101621924829&srcp=&tn=pc&idctag=tc&sids=10007_10407_10801_10607_10500_10702_10919_10911_11006_10924_10904_10018_10901_10942_10907_11012_10971_10968_10974_11031_12201_17851_17070_18013_18101_19107_18242_19201_17201_17203_18300_18401_18310_18315_18331_18412_19115_19123_19130_19147_9999&logid=3780406733&entrance=general&tpl_from=pc&page={}&'.format(i)

        html = get_page_html(url)
        real_urls = parse_result(html)
        for real_url in real_urls:
            real_url = real_url.replace('\\/', '/').replace('\\','//')
            content = get_image_content(real_url)
            save_pic(real_url, content)


# 函数入口
if __name__ == '__main__':
    main()











######  视频截图  ######
"""
def save_img():
    video_path = r'F:/20200917/'
    pic_path = r'F:/pic/'
    videos = os.listdir(video_path)
    for video_name in videos[:]:
        file_name = video_name[:-4]
        print(file_name)
        folder_name = file_name

        vc = cv2.VideoCapture(video_path + video_name)
        c = 1
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False

        timeF = 90

        while rval:
            rval, frame = vc.read()
            if (c % timeF == 0):
                cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame)

            c = c + 1
        vc.release()


save_img()
"""

"""
######  不同文件夹同名文件重命名  ######

srcpath = r'F:/戴口罩/111111/'
targetpath = r'F:/mask/'
# 累加，用于命名
i = 11001
# 返回指定路径下的文件和目录信息
pathDir = os.listdir(srcpath)
# 遍历
for allDir in pathDir:

    for j in os.listdir(os.path.join(srcpath, allDir)):
        # 路径拼接
        imgPath = os.path.join(srcpath, allDir, j)

        newtargetpath = targetpath + str(i).zfill(6) + '.jpg'  # zfill()向右对齐，用0补齐
        # 复制文件
        shutil.copyfile(imgPath, newtargetpath)
        # 打印被复制的文件
        print(imgPath)
        i += 1
"""
"""
# 把数据写入pkl文件，读取pkl文件

with open("C:/Users/dell/Desktop/peta_partition.pkl", "rb") as fo:
    while True:
        try:
            dict_data = pickle.load(fo, encoding='utf-8')
            print(dict_data.keys())
        except EOFError:
            break


for i in range(5):
    dict_data['weight_traintest'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_traintest'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_traintest'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_traintest'][i].append(random.uniform(0.3, 0.4))
    dict_data['weight_traintest'][i].append(random.uniform(0, 0.1))
    dict_data['weight_trainval'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_trainval'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_trainval'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_trainval'][i].append(random.uniform(0.3, 0.4))
    dict_data['weight_trainval'][i].append(random.uniform(0, 0.1))
    dict_data['weight_train'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_train'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_train'][i].append(random.uniform(0.1, 0.2))
    dict_data['weight_train'][i].append(random.uniform(0.3, 0.4))
    dict_data['weight_train'][i].append(random.uniform(0, 0.1))


for i in range(5):
    for j in range(19100, 21001):
        dict_data['train'][i].append(j)
    for j in range(21456, 22101):
        dict_data['train'][i].append(j)
    for j in range(22345, 23001):
        dict_data['train'][i].append(j)
    for j in range(23490, 23566):
        dict_data['train'][i].append(j)

    # val + traintest = 2052 + 17950 = 20002
    for j in range(19000, 19200):
        dict_data['val'][i].append(j)
    for j in range(21186, 21300):
        dict_data['val'][i].append(j)
    for j in range(22207, 22400):
        dict_data['val'][i].append(j)
    for j in range(23474, 23490):
        dict_data['val'][i].append(j)

    for j in range(19200, 21186):
        dict_data['traintest'][i].append(j)
    for j in range(21300, 22207):
        dict_data['traintest'][i].append(j)
    for j in range(22400, 23474):
        dict_data['traintest'][i].append(j)
    for j in range(23490, 23566):
        dict_data['traintest'][i].append(j)

    # trainval + test = 12095 + 7907 = 20002
    for j in range(19000, 20000):
        dict_data['trainval'][i].append(j)
    for j in range(21186, 22000):
        dict_data['trainval'][i].append(j)
    for j in range(22207, 23000):
        dict_data['trainval'][i].append(j)
    for j in range(23474, 23500):
        dict_data['trainval'][i].append(j)

    for j in range(20000, 21186):
        dict_data['test'][i].append(j)
    for j in range(22000, 22207):
        dict_data['test'][i].append(j)
    for j in range(23000, 23474):
        dict_data['test'][i].append(j)
    for j in range(23500, 23567):
        dict_data['test'][i].append(j)


# for i in range(19001, 23569):
#     dict_data['image'].append(str(i)+'.png')

# dict_data['att'].append('[1, 0, 0, 0]')
# dict_data['att_name'].append('personStanding')
# dict_data['att_name'].append('personSitting')
# dict_data['att_name'].append('personLieProne')
# dict_data['att_name'].append('personLying')
# dict_data['att_name'].append('personSquatting')
# for i in range(0, 19000):
#     dict_data['att'][i].append(1)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)

# for i in range(19000, 21186):
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(1)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)

# for i in range(21186, 22207):
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(1)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)

# for i in range(22207, 23474):
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(1)
#     dict_data['att'][i].append(0)

# for i in range(23474, 23568):
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(0)
#     dict_data['att'][i].append(1)

# for i in range(19001, 23569):
#     a = []
#     dict_data['att'].append(a)
# for j in range(19000, 23568):
#     for i in range(105):
#         dict_data['att'][j].append(-1)


with open("C:/Users/dell/Desktop/peta_partition.pkl", "wb") as fo:
    pickle.dump(dict_data, fo)
    fo.close()

with open("C:/Users/dell/Desktop/peta_partition.pkl", "rb") as fo:
    while True:
        try:
            dict_data = pickle.load(fo, encoding='utf-8')
            # print(len(dict_data['trainval'][0]))  # 11400
            # print(len(dict_data['train'][0]))  # 9500
            # print(len(dict_data['val'][0]))  # 1900
            # print(len(dict_data['test'][0]))  # 7600
            # print(len(dict_data['traintest'][0]))  # 17100
            # print(len(dict_data['att']))
            # print(dict_data['selected_attribute'])
            # print(dict_data['weight_trainval'][0])
            # print(dict_data['weight_trainval'][1])
            # print(dict_data['weight_trainval'][2])
            # print(dict_data['weight_trainval'][3])
            # print(dict_data['weight_trainval'][4])
            # print(len(dict_data['att']))
            # print(dict_data['att'][18400])

        except EOFError:
            break
"""

"""
# pkl2txt
import sys
sys.getdefaultencoding()
import pickle
import numpy as np
np.set_printoptions(threshold=1000000000000000)
path = 'C:/Users/dell/Desktop/peta_dataset_bak.pkl'
file = open(path,'rb')
inf = pickle.load(file,encoding='iso-8859-1')       #读取pkl文件的内容
print(inf)
#fr.close()
inf=str(inf)
obj_path = 'C:/Users/dell/Desktop/1.txt'
ft = open(obj_path, 'w')
ft.write(inf)
"""

"""
# 图片重命名
file = 'D:/baiduimage\get-down-cut/11111/nomask'
j = 0
for i in os.listdir(file):
    if (i.endswith('.jpg')):
        new_name = 'no_mask_' + str(j) + '.jpg'
        os.chdir(file)  #没有这一步的话会报 FileNotFoundError  的错误
        os.rename(i, new_name)
        j = j + 1
        print(i)
    else:
        print("no")
"""

# python-----截取xml文件画框的图片并保存
"""
# from __future__ import division

ImgPath = r'F:/mask/havemask/'
AnnoPath = r'D:/VOC2028/facemask-xml/'
ProcessedPath = r'D:/baiduimage/get-down-cut/2222/'

imagelist = os.listdir(ImgPath)

for image in imagelist:
    image_pre, ext = os.path.splitext(image)
    imgfile = ImgPath + image
    print(imgfile)
    if not os.path.exists(AnnoPath + image_pre + '.xml' ):
        continue
    xmlfile = AnnoPath + image_pre + '.xml'
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    filenamelist = annotation.getElementsByTagName('filename')
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    i = 1
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        savepath = ProcessedPath + objectname
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        bndbox = objects.getElementsByTagName('bndbox')
        cropboxes = []
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            w = x2 - x1
            h = y2 - y1
            obj = np.array([x1,y1,x2,y2])
            shift = np.array([[1,1,1,1]])
            XYmatrix = np.tile(obj,(1,1))
            cropboxes = XYmatrix * shift
            img = Image.open(imgfile)
            for cropbox in cropboxes:
                cropedimg = img.crop(cropbox)
                # cropedimg = img.convert('RGB')
                cropedimg.save(savepath + '/' + image_pre + '_' + str(i) + '.png')
                i += 1
"""

"""
# json to xml
def jsonToXml(json_str):
    try:
        xml_str=""
        xml_str = xmltodict.unparse(json_str, encoding='utf-8')
    except:
        xml_str = xmltodict.unparse({'request': json_str}, encoding='utf-8')
    finally:
        return xml_str

def json_to_xml(json_path,xml_path):
    if(os.path.exists(xml_path)==False):
        os.makedirs(xml_path)
    dir = os.listdir(json_path)
    for file in dir:
        file_list=file.split(".")
        with open(os.path.join(json_path,file), 'r') as load_f:
            load_dict = json.load(load_f)
        json_result = jsonToXml(load_dict)
        f = open(os.path.join(xml_path,file_list[0]+".xml"), 'w', encoding="UTF-8")
        f.write(json_result)
        f.close()

if __name__ == '__main__':
    json_path = r"C:/Users/dell/Desktop/111"  #该目录为存放json文件的路径  ps:目录中只能存放json文件
    xml_path=r"C:/Users/dell/Desktop/222"   #该目录为放xml文件的路径
    json_to_xml(json_path, xml_path)
"""

"""
# 文件名称转json
def read_directory(path, result):
    paths = os.listdir(path)
    for i, item in enumerate(paths):
        sub_path = os.path.join(path, item)
        if os.path.isdir(sub_path):
            result[item] = {}
            read_directory(sub_path, result[item])
        else:
            result[item] = item

if __name__ == '__main__':
    fpath = r'D:\baiduimage\get-down-cut\11111\nomask'
    filename = r'D:\识别代码\code\mask\json_res.json'
    result = {}
    read_directory(fpath, result)
    json_res = json.dumps(result, indent=2)
    print(json_res)
    with open(filename, 'w') as fp:
        fp.write(json_res)


# 修改json的value值
filename = r'D:\识别代码\code\mask\json_res.json'
with open(filename, 'r')as f:
    data = json.load(f)
    for key in data:
        data[key] = 0

tempfile = os.path.join(os.path.dirname(filename), str(uuid.uuid4()))
with open(tempfile, 'w') as f:
    json.dump(data, f, indent=4)
"""
"""
filename = r'D:\识别代码\code\mask\json\json_res.json'
filename_1 = r'D:\识别代码\code\mask\json\json_res-1.json'
resfile = r'D:\识别代码\code\mask\json\res.json'
file = json.load(open(filename, 'r'))
file_1 = json.load(open(filename_1, 'r'))

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

res = Merge(file, file_1)
with open(resfile, 'w') as fo:
    json.dump(res, fo)
"""

"""
# csv2txt
data = pd.read_csv('C:/Users/dell/Desktop/fairface_label_train_1.csv', encoding='utf-8')

with open('C:/Users/dell/Desktop/celebA_train.txt', 'a+', encoding='utf-8') as f:
    for line in data.values:
        
        if line[2] == 'Male':
            line[2] = 0
        else:
            line[2] = 1
        if line[3] == 'Indian' or line[3] == 'East Asian' or line[3] == 'Southeast Asian':
            line[3] = 0
        elif line[3] == 'Black':
            line[3] = 1
        elif line[3] == 'White' or line[3] == 'Middle Eastern':
            line[3] = 2
        else:
            line[3] = -1
        
        f.write((str(line[0]) + ' ' + str(-1) + ' ' + str(-1) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(-1)
                 + ' ' + str(-1) + ' ' + str(line[3]) + ' ' + str(-1) + ' ' + str(line[4]) + ' ' + str(-1) + ' '
                 + str(-1) + ' ' + '\n'))
"""

"""
def get_files(path, _ends=['*.json']):
    all_files = []
    for _end in _ends:
        files = glob.glob(os.path.join(path, _end))
        all_files.extend(files)
    file_num = len(all_files)
    return all_files, file_num


def get_text_mark(file_path):
    with open(file_path, 'r', encoding='utf-8') as fid:
        result_dict = json.load(fid)
        obj = result_dict['outputs']['object']
        names = file_path.split('\\')
        name = names[len(names)-1]
        file_name = name[:-5]
        all_text_mark = []
        all_text_mark.append(file_name+' ')
        all_text_mark.append(str(len(obj))+' ')
        for obj_item in obj:
            text = obj_item['name']
            try:
                coords = obj_item['polygon']
                try:
                    output_coord = [int(float(coords['x1'])), int(float(coords['y1'])), int(float(coords['x2']))
                        , int(float(coords['y2'])), int(float(coords['x3'])), int(float(coords['y3'])),
                                    int(float(coords['x4'])), int(float(coords['y4']))]
                except:
                    continue
            except:
                coords = obj_item['bndbox']
                try:
                    output_coord = [int(float(coords['xmin'])), int(float(coords['ymin'])),
                                    int(float(coords['xmax'])), int(float(coords['ymax']))]
                    print(output_coord)
                except:
                    continue
            output_text = text + ' 0.99 '
            for item in output_coord:
                output_text = output_text + str(item) + ' '
            all_text_mark.append(output_text)
        return all_text_mark


def write_to_txt(out_txt_path, one_file_all_mark):
    # windows
    with open(os.path.join(out_txt_path, file.split('\\')
                                         [-1].split('.')[0] + '.txt'), 'a+', encoding='utf-8') as fid:
        ##linux
        # with open(os.path.join(out_txt_path, file.split('/')
        #                                      [-1].split('.')[0] + '.txt'), 'a+', encoding='utf-8') as fid:
        for item in one_file_all_mark:
            fid.write(item)


if __name__ == "__main__":
    json_path = 'C:/Users/dell/Desktop'
    out_txt_path = 'C:/Users/dell/Desktop/test.txt'
    files, files_len = get_files(json_path)
    bar = tqdm(total=files_len)
    with open(out_txt_path, 'a+', encoding='utf-8') as fid:
        for file in files:
            bar.update(1)
            print(file)
            try:
                one_file_all_mark = get_text_mark(file)
            except:
                print(file)
                continue
            for item in one_file_all_mark:
                fid.write(item)
            fid.write('\n')
    bar.close()
"""

"""
def countnum(list):
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(len(list)):
        for j in range(len(list[i])):
            if list[i][j] == '1':
                count1 += 1
            elif list[i][j] == '0':
                count2 += 1
            else:
                count3 += 1
    return count1, count2, count3

def main():
    f = codecs.open('C:/Users/dell/Desktop/train.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
    line = f.readline()  # 以行的形式进行读取文件
    personalLess30list = []
    personalLess45list = []
    personalLess60list = []
    personalLarger60list = []
    carryingBackpacklist = []
    carryingOtherlist = []
    lowerBodyCasuallist = []
    upperBodyCasuallist = []
    lowerBodyFormallist = []
    upperBodyFormallist = []
    accessoryHatlist = []
    upperBodyJacketlist = []
    lowerBodyJeanslist = []
    footwearLeatherShoeslist = []
    upperBodyLogolist = []
    hairLonglist = []
    personalMalelist = []
    carryingMessengerBaglist = []
    accessoryMufflerlist = []
    accessoryNothinglist = []
    carryingNothinglist = []
    upperBodyPlaidlist = []
    carryingPlasticBagslist = []
    footwearSandalslist = []
    footwearShoeslist = []
    lowerBodyShortslist = []
    upperBodyShortSleevelist = []
    lowerBodyShortSkirtlist = []
    footwearSneakerlist = []
    upperBodyThinStripeslist = []
    accessorySunglasseslist = []
    lowerBodyTrouserslist = []
    upperBodyTshirtlist = []
    upperBodyOtherlist = []
    upperBodyVNecklist = []
    upperBodyBlacklist = []
    upperBodyBluelist = []
    upperBodyBrownlist = []
    upperBodyGreenlist = []
    upperBodyGreylist = []
    upperBodyOrangelist = []
    upperBodyPinklist = []
    upperBodyPurplelist = []
    upperBodyRedlist = []
    upperBodyWhitelist = []
    upperBodyYellowlist = []
    lowerBodyBlacklist = []
    lowerBodyBluelist = []
    lowerBodyBrownlist = []
    lowerBodyGreenlist = []
    lowerBodyGreylist = []
    lowerBodyOrangelist = []
    lowerBodyPinklist = []
    lowerBodyPurplelist = []
    lowerBodyRedlist = []
    lowerBodyWhitelist = []
    lowerBodyYellowlist = []
    hairBlacklist = []
    hairBluelist = []
    hairBrownlist = []
    hairGreenlist = []
    hairGreylist = []
    hairOrangelist = []
    hairPinklist = []
    hairPurplelist = []
    hairRedlist = []
    hairWhitelist = []
    hairYellowlist = []
    footwearBlacklist = []
    footwearBluelist = []
    footwearBrownlist = []
    footwearGreenlist = []
    footwearGreylist = []
    footwearOrangelist = []
    footwearPinklist = []
    footwearPurplelist = []
    footwearRedlist = []
    footwearWhitelist = []
    footwearYellowlist = []
    accessoryHeadphonelist = []
    personalLess15list = []
    carryingBabyBuggylist = []
    hairBaldlist = []
    footwearBootslist = []
    lowerBodyCaprilist = []
    carryingShoppingTrolist = []
    carryingUmbrellalist = []
    personalFemalelist = []
    carryingFolderlist = []
    accessoryHairBandlist = []
    lowerBodyHotPantslist = []
    accessoryKerchieflist = []
    lowerBodyLongSkirtlist = []
    upperBodyLongSleevelist = []
    lowerBodyPlaidlist = []
    lowerBodyThinStripeslist = []
    carryingLuggageCaselist = []
    upperBodyNoSleevelist = []
    hairShortlist = []
    footwearStockinglist = []
    upperBodySuitlist = []
    carryingSuitcaselist = []
    lowerBodySuitslist = []
    upperBodySweaterlist = []
    upperBodyThickStripeslist = []
    personStandinglist = []
    personSittinglist = []
    personLiePronelist = []
    personLyinglist = []
    personSquattinglist = []
    angleFrontlist = []
    angleSidelist = []
    angleBacklist = []
    hasMasklist = []
    upperBodySolidColorlist = []
    upperBodyDesignPatternlist = []
    upperBodyJointlist = []
    lowerBodySolidColorlist = []
    lowerBodyDesignPatternlist = []
    upperBodyDarkgraylist = []
    upperBodyJewelryBluelist = []
    upperBodyGrayishBluelist = []
    upperBodyLightBluelist = []
    upperBodyKhakilist = []
    upperBodyCyanlist = []
    upperBodySilverlist = []
    lowerBodyDarkgraylist = []
    lowerBodyJewelryBluelist = []
    lowerBodyGrayishBluelist = []
    lowerBodyLightBluelist = []
    lowerBodyKhakilist = []
    lowerBodyCyanlist = []
    lowerBodySilverlist = []
    while line:
        a = line.split()
        personalLess30 = a[1:2]
        personalLess30list.append(personalLess30)  # 将其添加在列表之中
        personalLess45 = a[2:3]
        personalLess45list.append(personalLess45)  # 将其添加在列表之中
        personalLess60 = a[3:4]
        personalLess60list.append(personalLess60)  # 将其添加在列表之中
        personalLarger60 = a[4:5]
        personalLarger60list.append(personalLarger60)
        carryingBackpack = a[5:6]
        carryingBackpacklist.append(carryingBackpack)
        carryingOther = a[6:7]
        carryingOtherlist.append(carryingOther)
        lowerBodyCasual = a[7:8]
        lowerBodyCasuallist.append(lowerBodyCasual)
        upperBodyCasual = a[8:9]
        upperBodyCasuallist.append(upperBodyCasual)
        lowerBodyFormal = a[9:10]
        lowerBodyFormallist.append(lowerBodyFormal)
        upperBodyFormal = a[10:11]
        upperBodyFormallist.append(upperBodyFormal)
        accessoryHat = a[11:12]
        accessoryHatlist.append(accessoryHat)
        upperBodyJacket = a[12:13]
        upperBodyJacketlist.append(upperBodyJacket)
        lowerBodyJeans = a[13:14]
        lowerBodyJeanslist.append(lowerBodyJeans)
        footwearLeatherShoes = a[14:15]
        footwearLeatherShoeslist.append(footwearLeatherShoes)
        upperBodyLogo = a[15:16]
        upperBodyLogolist.append(upperBodyLogo)
        hairLong = a[16:17]
        hairLonglist.append(hairLong)
        personalMale = a[17:18]
        personalMalelist.append(personalMale)
        carryingMessengerBag = a[18:19]
        carryingMessengerBaglist.append(carryingMessengerBag)
        accessoryMuffler = a[19:20]
        accessoryMufflerlist.append(accessoryMuffler)
        accessoryNothing = a[20:21]
        accessoryNothinglist.append(accessoryNothing)
        carryingNothing = a[21:22]
        carryingNothinglist.append(carryingNothing)
        upperBodyPlaid = a[22:23]
        upperBodyPlaidlist.append(upperBodyPlaid)
        carryingPlasticBags = a[23:24]
        carryingPlasticBagslist.append(carryingPlasticBags)
        footwearSandals = a[24:25]
        footwearSandalslist.append(footwearSandals)
        footwearShoes = a[25:26]
        footwearShoeslist.append(footwearShoes)
        lowerBodyShorts = a[26:27]
        lowerBodyShortslist.append(lowerBodyShorts)
        upperBodyShortSleeve = a[27:28]
        upperBodyShortSleevelist.append(upperBodyShortSleeve)
        lowerBodyShortSkirt = a[28:29]
        lowerBodyShortSkirtlist.append(lowerBodyShortSkirt)
        footwearSneaker = a[29:30]
        footwearSneakerlist.append(footwearSneaker)
        upperBodyThinStripes = a[30:31]
        upperBodyThinStripeslist.append(upperBodyThinStripes)
        accessorySunglasses = a[31:32]
        accessorySunglasseslist.append(accessorySunglasses)
        lowerBodyTrousers = a[32:33]
        lowerBodyTrouserslist.append(lowerBodyTrousers)
        upperBodyTshirt = a[33:34]
        upperBodyTshirtlist.append(upperBodyTshirt)
        upperBodyOther = a[34:35]
        upperBodyOtherlist.append(upperBodyOther)
        upperBodyVNeck = a[35:36]
        upperBodyVNecklist.append(upperBodyVNeck)
        upperBodyBlack = a[36:37]
        upperBodyBlacklist.append(upperBodyBlack)
        upperBodyBlue = a[37:38]
        upperBodyBluelist.append(upperBodyBlue)
        upperBodyBrown = a[38:39]
        upperBodyBrownlist.append(upperBodyBrown)
        upperBodyGreen = a[39:40]
        upperBodyGreenlist.append(upperBodyGreen)
        upperBodyGrey = a[40:41]
        upperBodyGreylist.append(upperBodyGrey)
        upperBodyOrange = a[41:42]
        upperBodyOrangelist.append(upperBodyOrange)
        upperBodyPink = a[42:43]
        upperBodyPinklist.append(upperBodyPink)
        upperBodyPurple = a[43:44]
        upperBodyPurplelist.append(upperBodyPurple)
        upperBodyRed = a[44:45]
        upperBodyRedlist.append(upperBodyRed)
        upperBodyWhite = a[45:46]
        upperBodyWhitelist.append(upperBodyWhite)
        upperBodyYellow = a[46:47]
        upperBodyYellowlist.append(upperBodyYellow)
        lowerBodyBlack = a[47:48]
        lowerBodyBlacklist.append(lowerBodyBlack)
        lowerBodyBlue = a[48:49]
        lowerBodyBluelist.append(lowerBodyBlue)
        lowerBodyBrown = a[49:50]
        lowerBodyBrownlist.append(lowerBodyBrown)
        lowerBodyGreen = a[50:51]
        lowerBodyGreenlist.append(lowerBodyGreen)
        lowerBodyGrey = a[51:52]
        lowerBodyGreylist.append(lowerBodyGrey)
        lowerBodyOrange = a[52:53]
        lowerBodyOrangelist.append(lowerBodyOrange)
        lowerBodyPink = a[53:54]
        lowerBodyPinklist.append(lowerBodyPink)
        lowerBodyPurple = a[54:55]
        lowerBodyPurplelist.append(lowerBodyPurple)
        lowerBodyRed = a[55:56]
        lowerBodyRedlist.append(lowerBodyRed)
        lowerBodyWhite = a[56:57]
        lowerBodyWhitelist.append(lowerBodyWhite)
        lowerBodyYellow = a[57:58]
        lowerBodyYellowlist.append(lowerBodyYellow)
        hairBlack = a[58:59]
        hairBlacklist.append(hairBlack)
        hairBlue = a[59:60]
        hairBluelist.append(hairBlue)
        hairBrown = a[60:61]
        hairBrownlist.append(hairBrown)
        hairGreen = a[61:62]
        hairGreenlist.append(hairGreen)
        hairGrey = a[62:63]
        hairGreylist.append(hairGrey)
        hairOrange = a[63:64]
        hairOrangelist.append(hairOrange)
        hairPink = a[64:65]
        hairPinklist.append(hairPink)
        hairPurple = a[65:66]
        hairPurplelist.append(hairPurple)
        hairRed = a[66:67]
        hairRedlist.append(hairRed)
        hairWhite = a[67:68]
        hairWhitelist.append(hairWhite)
        hairYellow = a[68:69]
        hairYellowlist.append(hairYellow)
        footwearBlack = a[69:70]
        footwearBlacklist.append(footwearBlack)
        footwearBlue = a[70:71]
        footwearBluelist.append(footwearBlue)
        footwearBrown = a[71:72]
        footwearBrownlist.append(footwearBrown)
        footwearGreen = a[72:73]
        footwearGreenlist.append(footwearGreen)
        footwearGrey = a[73:74]
        footwearGreylist.append(footwearGrey)
        footwearOrange = a[74:75]
        footwearOrangelist.append(footwearOrange)
        footwearPink = a[75:76]
        footwearPinklist.append(footwearPink)
        footwearPurple = a[76:77]
        footwearPurplelist.append(footwearPurple)
        footwearRed = a[77:78]
        footwearRedlist.append(footwearRed)
        footwearWhite = a[78:79]
        footwearWhitelist.append(footwearWhite)
        footwearYellow = a[79:80]
        footwearYellowlist.append(footwearYellow)
        accessoryHeadphone = a[80:81]
        accessoryHeadphonelist.append(accessoryHeadphone)
        personalLess15 = a[81:82]
        personalLess15list.append(personalLess15)
        carryingBabyBuggy = a[82:83]
        carryingBabyBuggylist.append(carryingBabyBuggy)
        hairBald = a[83:84]
        hairBaldlist.append(hairBald)
        footwearBoots = a[84:85]
        footwearBootslist.append(footwearBoots)
        lowerBodyCapri = a[85:86]
        lowerBodyCaprilist.append(lowerBodyCapri)
        carryingShoppingTro = a[86:87]
        carryingShoppingTrolist.append(carryingShoppingTro)
        carryingUmbrella = a[87:88]
        carryingUmbrellalist.append(carryingUmbrella)
        personalFemale = a[88:89]
        personalFemalelist.append(personalFemale)
        carryingFolder = a[89:90]
        carryingFolderlist.append(carryingFolder)
        accessoryHairBand = a[90:91]
        accessoryHairBandlist.append(accessoryHairBand)
        lowerBodyHotPants = a[91:92]
        lowerBodyHotPantslist.append(lowerBodyHotPants)
        accessoryKerchief = a[92:93]
        accessoryKerchieflist.append(accessoryKerchief)
        lowerBodyLongSkirt = a[93:94]
        lowerBodyLongSkirtlist.append(lowerBodyLongSkirt)
        upperBodyLongSleeve = a[94:95]
        upperBodyLongSleevelist.append(upperBodyLongSleeve)
        lowerBodyPlaid = a[95:96]
        lowerBodyPlaidlist.append(lowerBodyPlaid)
        lowerBodyThinStripes = a[96:97]
        lowerBodyThinStripeslist.append(lowerBodyThinStripes)
        carryingLuggageCase = a[97:98]
        carryingLuggageCaselist.append(carryingLuggageCase)
        upperBodyNoSleeve = a[98:99]
        upperBodyNoSleevelist.append(upperBodyNoSleeve)
        hairShort = a[99:100]
        hairShortlist.append(hairShort)
        footwearStocking = a[100:101]
        footwearStockinglist.append(footwearStocking)
        upperBodySuit = a[101:102]
        upperBodySuitlist.append(upperBodySuit)
        carryingSuitcase = a[102:103]
        carryingSuitcaselist.append(carryingSuitcase)
        lowerBodySuits = a[103:104]
        lowerBodySuitslist.append(lowerBodySuits)
        upperBodySweater = a[104:105]
        upperBodySweaterlist.append(upperBodySweater)
        upperBodyThickStripes = a[105:106]
        upperBodyThickStripeslist.append(upperBodyThickStripes)
        personStanding = a[106:107]
        personStandinglist.append(personStanding)
        personSitting = a[107:108]
        personSittinglist.append(personSitting)
        personLieProne = a[108:109]
        personLiePronelist.append(personLieProne)
        personLying = a[109:110]
        personLyinglist.append(personLying)
        personSquatting = a[110:111]
        personSquattinglist.append(personSquatting)
        angleFront = a[111:112]
        angleFrontlist.append(angleFront)
        angleSide = a[112:113]
        angleSidelist.append(angleSide)
        angleBack = a[113:114]
        angleBacklist.append(angleBack)
        hasMask = a[114:115]
        hasMasklist.append(hasMask)
        upperBodySolidColor = a[115:116]
        upperBodySolidColorlist.append(upperBodySolidColor)
        upperBodyDesignPattern = a[116:117]
        upperBodyDesignPatternlist.append(upperBodyDesignPattern)
        upperBodyJoint = a[117:118]
        upperBodyJointlist.append(upperBodyJoint)
        lowerBodySolidColor = a[118:119]
        lowerBodySolidColorlist.append(lowerBodySolidColor)
        lowerBodyDesignPattern = a[119:120]
        lowerBodyDesignPatternlist.append(lowerBodyDesignPattern)
        upperBodyDarkgray = a[120:121]
        upperBodyDarkgraylist.append(upperBodyDarkgray)
        upperBodyJewelryBlue = a[121:122]
        upperBodyJewelryBluelist.append(upperBodyJewelryBlue)
        upperBodyGrayishBlue = a[122:123]
        upperBodyGrayishBluelist.append(upperBodyGrayishBlue)
        upperBodyLightBlue = a[123:124]
        upperBodyLightBluelist.append(upperBodyLightBlue)
        upperBodyKhaki = a[124:125]
        upperBodyKhakilist.append(upperBodyKhaki)
        upperBodyCyan = a[125:126]
        upperBodyCyanlist.append(upperBodyCyan)
        upperBodySilver = a[126:127]
        upperBodySilverlist.append(upperBodySilver)
        lowerBodyDarkgray = a[127:128]
        lowerBodyDarkgraylist.append(lowerBodyDarkgray)
        lowerBodyJewelryBlue = a[128:129]
        lowerBodyJewelryBluelist.append(lowerBodyJewelryBlue)
        lowerBodyGrayishBlue = a[129:130]
        lowerBodyGrayishBluelist.append(lowerBodyGrayishBlue)
        lowerBodyLightBlue = a[130:131]
        lowerBodyLightBluelist.append(lowerBodyLightBlue)
        lowerBodyKhaki = a[131:132]
        lowerBodyKhakilist.append(lowerBodyKhaki)
        lowerBodyCyan = a[132:133]
        lowerBodyCyanlist.append(lowerBodyCyan)
        lowerBodySilver = a[133:134]
        lowerBodySilverlist.append(lowerBodySilver)

        line = f.readline()
    f.close()

    i, j, k = countnum(personalLess30list)
    print('personalLess30 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personalLess45list)
    print('personalLess45 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personalLess60list)
    print('personalLess60 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personalLarger60list)
    print('personalLarger60 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingBackpacklist)
    print('carryingBackpack 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingOtherlist)
    print('carryingOther 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyCasuallist)
    print('lowerBodyCasual 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyCasuallist)
    print('upperBodyCasual 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyFormallist)
    print('lowerBodyFormal 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyFormallist)
    print('upperBodyFormal 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(accessoryHatlist)
    print('accessoryHat 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyJacketlist)
    print('upperBodyJacket 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyJeanslist)
    print('lowerBodyJeans 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearLeatherShoeslist)
    print('footwearLeatherShoes 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyLogolist)
    print('upperBodyLogo 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairLonglist)
    print('hairLong 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personalMalelist)
    print('personalMale 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingMessengerBaglist)
    print('carryingMessengerBag 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(accessoryMufflerlist)
    print('accessoryMuffler 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(accessoryNothinglist)
    print('accessoryNothing 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingNothinglist)
    print('carryingNothing 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyPlaidlist)
    print('upperBodyPlaid 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingPlasticBagslist)
    print('carryingPlasticBags 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearSandalslist)
    print('footwearSandals 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearShoeslist)
    print('footwearShoes 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyShortslist)
    print('lowerBodyShorts 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyShortSleevelist)
    print('upperBodyShortSleeve 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyShortSkirtlist)
    print('lowerBodyShortSkirt 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearSneakerlist)
    print('footwearSneaker 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyThinStripeslist)
    print('upperBodyThinStripes 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(accessorySunglasseslist)
    print('accessorySunglasses 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyTrouserslist)
    print('lowerBodyTrousers 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyTshirtlist)
    print('upperBodyTshirt 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyOtherlist)
    print('upperBodyOther 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyVNecklist)
    print('upperBodyVNeck 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyBlacklist)
    print('upperBodyBlack 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyBluelist)
    print('upperBodyBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyBrownlist)
    print('upperBodyBrown 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyGreenlist)
    print('upperBodyGreen 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyGreylist)
    print('upperBodyGrey 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyOrangelist)
    print('upperBodyOrange 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyPinklist)
    print('upperBodyPink 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyPurplelist)
    print('upperBodyPurple 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyRedlist)
    print('upperBodyRed 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyWhitelist)
    print('upperBodyWhite 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyYellowlist)
    print('upperBodyYellow 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyBlacklist)
    print('lowerBodyBlack 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyBluelist)
    print('lowerBodyBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyBrownlist)
    print('lowerBodyBrown 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyGreenlist)
    print('lowerBodyGreen 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyGreylist)
    print('lowerBodyGrey 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyOrangelist)
    print('lowerBodyOrange 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyPinklist)
    print('lowerBodyPink 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyPurplelist)
    print('lowerBodyPurple 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyRedlist)
    print('lowerBodyRed 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyWhitelist)
    print('lowerBodyWhite 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyYellowlist)
    print('lowerBodyYellow 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairBlacklist)
    print('hairBlack 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairBluelist)
    print('hairBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairBrownlist)
    print('hairBrown 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairGreenlist)
    print('hairGreen 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairGreylist)
    print('hairGrey 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairOrangelist)
    print('hairOrange 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairPinklist)
    print('hairPink 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairPurplelist)
    print('hairPurple 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairRedlist)
    print('hairRed 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairWhitelist)
    print('hairWhite 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairYellowlist)
    print('hairYellow 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearBlacklist)
    print('footwearBlack 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearBluelist)
    print('footwearBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearBrownlist)
    print('footwearBrown 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearGreenlist)
    print('footwearGreen 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearGreylist)
    print('footwearGrey 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearOrangelist)
    print('footwearOrange 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearPinklist)
    print('footwearPink 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearPurplelist)
    print('footwearPurple 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearRedlist)
    print('footwearRed 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearWhitelist)
    print('footwearWhite 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearYellowlist)
    print('footwearYellow 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(accessoryHeadphonelist)
    print('accessoryHeadphone 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personalLess15list)
    print('personalLess15 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingBabyBuggylist)
    print('carryingBabyBuggy 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairBaldlist)
    print('hairBald 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearBootslist)
    print('footwearBoots 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyCaprilist)
    print('lowerBodyCapri 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingShoppingTrolist)
    print('carryingShoppingTro 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingUmbrellalist)
    print('carryingUmbrella 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personalFemalelist)
    print('personalFemale 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingFolderlist)
    print('carryingFolder 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(accessoryHairBandlist)
    print('accessoryHairBand 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyHotPantslist)
    print('lowerBodyHotPants 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(accessoryKerchieflist)
    print('accessoryKerchief 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyLongSkirtlist)
    print('lowerBodyLongSkirt 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyLongSleevelist)
    print('upperBodyLongSleeve 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyPlaidlist)
    print('lowerBodyPlaid 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyThinStripeslist)
    print('lowerBodyThinStripes 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingLuggageCaselist)
    print('carryingLuggageCase 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyNoSleevelist)
    print('upperBodyNoSleeve 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hairShortlist)
    print('hairShort 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(footwearStockinglist)
    print('footwearStocking 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodySuitlist)
    print('upperBodySuit 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(carryingSuitcaselist)
    print('carryingSuitcase 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodySuitslist)
    print('lowerBodySuits 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodySweaterlist)
    print('upperBodySweater 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyThickStripeslist)
    print('upperBodyThickStripes 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personStandinglist)
    print('personStanding 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personSittinglist)
    print('personSitting 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personLiePronelist)
    print('personLieProne 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personLyinglist)
    print('personLying 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(personSquattinglist)
    print('personSquatting 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(angleFrontlist)
    print('angleFront 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(angleSidelist)
    print('angleSide 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(angleBacklist)
    print('angleBack 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(hasMasklist)
    print('hasMask 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodySolidColorlist)
    print('upperBodySolidColor 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyDesignPatternlist)
    print('upperBodyDesignPattern 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyJointlist)
    print('upperBodyJoint 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodySolidColorlist)
    print('lowerBodySolidColor 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyDesignPatternlist)
    print('lowerBodyDesignPattern 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyDarkgraylist)
    print('upperBodyDarkgray 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyJewelryBluelist)
    print('upperBodyJewelryBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyGrayishBluelist)
    print('upperBodyGrayishBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyLightBluelist)
    print('upperBodyLightBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyKhakilist)
    print('upperBodyKhaki 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodyCyanlist)
    print('upperBodyCyan 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(upperBodySilverlist)
    print('upperBodySilver 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyDarkgraylist)
    print('lowerBodyDarkgray 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyJewelryBluelist)
    print('lowerBodyJewelryBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyGrayishBluelist)
    print('lowerBodyGrayishBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyLightBluelist)
    print('lowerBodyLightBlue 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyKhakilist)
    print('lowerBodyKhaki 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodyCyanlist)
    print('lowerBodyCyan 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
    i, j, k = countnum(lowerBodySilverlist)
    print('lowerBodySilver 变量为1:%d，变量为0:%d，变量为-1:%d' % (i, j, k))
if __name__ == '__main__':
    main()
"""
# path_out = 'C:/Users/dell/Desktop/celebA_test_new.txt'   # 新的txt文件
# t = ''
# with open(path_out, 'w+') as f_out:
#     for i in list9:
#         for j in range(len(list9[0])):
#             t = t + str(i[j])
#         f_out.write(t)
#         f_out.write('\n')
#         t = ''


"""
attr_path = 'C:/Users/dell/Desktop/111'
val_list = [[], []]
test_list = [[], []]
for root_dir, dirs, names in os.walk(attr_path):
    for n in names:
        path = os.path.join(root_dir, n)
        with open(path, 'rb') as f:
            # print(f)
            data = pickle.load(f)
            if data['age']:
                if len(val_list[1]) < 100000:
                    val_list.append(path)
                    val_list.append(data['age'])
                    val_list.append(data['attractive'])
                    val_list.append(data['gender'])
                    val_list.append(data['eyeglass'])
                    val_list.append(data['eyeOpen'])
                    val_list.append(data['sunglass'])
                    val_list.append(data['smile'])
                    val_list.append(data['mask'])
                    val_list.append(data['mouthOpen'])
                    val_list.append(data['beard'])
                    val_list.append(data['race'])
                    val_list.append('...')
            else:
                if len(val_list[0]) < 100000:
                    val_list[0].append(path)
                    continue
                elif len(test_list[0]) < 100000:
                    test_list[0].append(path)
                    continue

test_path = './test_label.txt'
val_path = './val_label.txt'
print(val_list)


t = ''
with open(val_path, 'w+') as f_out:
    for i in val_list:
        for j in range(val_list)
            t = t + str(i[j])
        f_out.write(t+'\n')
        t = ''

with open(test_path, 'w') as f:
    for list in test_list:
        t =''
        for p in list:
            with open(p, 'rb') as f_t:
                data = pickle.load(f_t.read())
                #print(test_path, data['beard'])
                t = str(test_path)+' '+str(data['age'])+' '+str(data['attractive'])+' '+str(data['gender'])

                f_t.write(t+'\n')
                t = ''

with open(val_path, 'w') as f:
    for list in val_list:
        for p in list:
            with open(p, 'rb') as f_t:
                data = pickle.load(f_t.read())
                #print(test_path, data['beard'])
                t = str(val_path)+' '+str(data['age'])+' '+str(data['attractive'])+' '+str(data['gender'])

                f_t.write(t+'\n')
                t = ''
            # f.write(p+'\n')

"""

##########################合并同一个文件夹下多个txt################

"""
def MergeTxt(filepath,outfile):
    k = open(filepath+outfile, 'a+')
    for parent, dirnames, filenames in os.walk(filepath):
        for filepath in filenames:
            ##########换行写入##################
            num = filepath.split('.')[-2]
            # txtPath = os.path.join(num, filepath)
            txtPath = os.path.join(parent, filepath)  # txtpath就是所有文件夹的路径
            f = open(txtPath)
            k.write(f.read())
    k.close()
    print("finished")

if __name__ == '__main__':
    filepath="F:/微信/企业微信/WXWork Files/File/2021-01/Mask_new_label/"
    outfile = "result.txt"
    MergeTxt(filepath, outfile)
    time2 = time.time()
    print(u'总共耗时：' + str(time2 - time1) + 's')
"""

"""
import os.path  # 文件夹遍历函数
# 获取目标文件夹的路径
# filedir = F:/微信/企业微信/WXWork Files/File/2021-01/Mask_new_label/
# 获取当前文件夹中的文件名称列表
filenames = os.listdir(filedir)
# 打开当前目录下的result.txt文件，如果没有则创建
f = open('result.txt', 'w')
# 先遍历文件名
for filename in filenames:
    filepath = filedir+'/'+filename
    # 遍历单个文件，读取行数
    for line in open(filepath):
        f.writelines(filename + line)
    f.write('\n')
# 关闭文件
f.close()
"""

"""
def get_imglist(src, img_list):
    for file in os.listdir(src):
        cur_path = os.path.join(src,file)
        if os.path.isdir(cur_path):
            get_imglist(cur_path,img_list)
        else:
            img_list.append(cur_path)
    return img_list


def add_img_path(img_path,label_path):
    file_list_1 = []
    file_list_label = get_imglist(label_path, file_list_1)
    for file in file_list_label:
        img_name = file.split('/')[-1].replace('txt', '.png')
        img_add = img_path + img_name.replace('\\', '/')
        print("Processing image:", img_name)
        with open(file, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(img_add + ' ' + content)


def get_all_files(label_src):
    folder_list = os.listdir(label_src)
    valid_folder_list = []
    all_files_list = []
    for folder in folder_list:
        if 'labels_new_' in folder:
            valid_folder_list.append(folder)
    for folder in valid_folder_list:
        int_list = []
        file_list = get_imglist(label_src+'/'+folder, int_list)
        all_files_list += file_list
    return all_files_list


def data_split(files_all, src1, src2, src3, src4, src5):
    ratio_train = 0.9  # 训练集比例
    # ratio_trainval = 0.2  # 验证集比例
    ratio_val = 0.1  # 测试集比例
    assert (ratio_train + ratio_val) == 1.0, 'Total ratio Not equal to 1'  # 检查总比例是否等于1

    cnt_val = round(len(files_all) * ratio_val, 0)
    cnt_train = len(files_all) - cnt_val
    print("test Sample:" + str(cnt_val))
    print("train Sample:" + str(cnt_train))

    # 打乱文件列表
    np.random.seed(2)
    np.random.shuffle(files_all)

    train_label_list = []
    val_label_list = []
    print('processing train.txt')
    for i in range(int(cnt_train)):
        img_txt = files_all[i].split('/')[-1]
        if img_txt.split('\\')[0] == 'labels_new_VERI-Wild':
            label_img_path = src1 + img_txt.replace('labels_new_VERI-Wild\\', '')\
                .replace('.txt', '.jpg')\
                .replace('\\', '/')\
                .replace('_test', '')\
                .replace('_train', '')
        if img_txt.split('\\')[0] == 'labels_new_CVPR-VehicleReId2016':
            label_img_path = src2 + img_txt.replace('labels_new_CVPR-VehicleReId2016\\', '') \
                .replace('.txt', '.png') \
                .replace('\\', '/')
        if img_txt.split('\\')[0] == 'labels_new_VRIC':
            label_img_path = src3 + img_txt.replace('labels_new_VRIC\\', '') \
                .replace('.txt', '.jpg') \
                .replace('\\', '/')
        if img_txt.split('\\')[0] == 'labels_new_VeRi':
            label_img_path = src4 + img_txt.replace('labels_new_VeRi\\', '') \
                .replace('.txt', '.jpg') \
                .replace('\\', '/')
        if img_txt.split('\\')[0] == 'labels_new_VehicleID':
            label_img_path = src5 + img_txt.replace('labels_new_VehicleID\\', '') \
                .replace('.txt', '.jpg') \
                .replace('\\', '/')
        # 读取文件
        with open(files_all[i], 'r', encoding="utf-8") as file:
            label = file.read()
        label_img = label_img_path + ' ' + label
        train_label_list.append(label_img)
    with open("train.txt", "w") as f:
        f.write('\n'.join(train_label_list))

    print('processing val.txt')
    for i in range(int(cnt_train), int(cnt_train + cnt_val)):
        img_txt = files_all[i].split('/')[-1]
        if img_txt.split('\\')[0] == 'labels_new_VERI-Wild':
            label_img_path = src1 + img_txt.replace('labels_new_VERI-Wild\\', '') \
                .replace('.txt', '.jpg') \
                .replace('\\', '/') \
                .replace('_test', '') \
                .replace('_train', '')
        if img_txt.split('\\')[0] == 'labels_new_CVPR-VehicleReId2016':
            label_img_path = src2 + img_txt.replace('labels_new_CVPR-VehicleReId2016\\', '') \
                .replace('.txt', '.png') \
                .replace('\\', '/')
        if img_txt.split('\\')[0] == 'labels_new_VRIC':
            label_img_path = src3 + img_txt.replace('labels_new_VRIC\\', '') \
                .replace('.txt', '.jpg') \
                .replace('\\', '/')
        if img_txt.split('\\')[0] == 'labels_new_VeRi':
            label_img_path = src4 + img_txt.replace('labels_new_VeRi\\', '') \
                .replace('.txt', '.jpg') \
                .replace('\\', '/')
        if img_txt.split('\\')[0] == 'labels_new_VehicleID':
            label_img_path = src5 + img_txt.replace('labels_new_VehicleID\\', '') \
                .replace('.txt', '.jpg') \
                .replace('\\', '/')
        # 读取文件
        with open(files_all[i], 'r', encoding="utf-8") as file:
            label = file.read()
        label_img = label_img_path + ' ' + label
        val_label_list.append(label_img)
    with open("val.txt", "w") as f:
        f.write('\n'.join(val_label_list))
    print('done')


# 文件路径
label_file_src = 'D:/liyiman/project_znv/label_add_vehicle'
img_path_1 = "/data_01/data2/puheng/vehicle_attribute/dataset/VERI-Wild/images/"
img_path_2 = "/data_01/data2/zhanghao/data/vehicle/CVPR-VehicleReId2016/"
img_path_3 = "/data_01/data2/zhanghao/data/vehicle/VRIC/"
img_path_4 = "/data_01/data2/zhanghao/data/vehicle/VeRi/"
img_path_5 = "/data_01/data2/zhanghao/data/vehicle/VehicleID/"

all_files = get_all_files(label_file_src)
data_split(all_files, img_path_1, img_path_2, img_path_3, img_path_4, img_path_5)
"""
"""
pos_ratio = torch.Tensor([4.9228e-01, 3.3386e-01, 1.0219e-01, 6.2368e-02, 1.9632e-01, 1.9807e-01,
                          8.6053e-01, 8.5193e-01, 1.3763e-01, 1.3491e-01, 1.0570e-01, 6.8421e-02,
                          3.0702e-01, 2.9939e-01, 4.0965e-02, 2.3763e-01, 5.5035e-01, 2.9333e-01,
                          8.4386e-02, 7.4526e-01, 2.7649e-01, 2.6491e-02, 7.6140e-02, 2.1404e-02,
                          3.6211e-01, 3.5000e-02, 1.4079e-01, 4.5000e-02, 2.1325e-01, 1.7544e-02,
                          2.9298e-02, 5.1211e-01, 8.4211e-02, 4.5614e-01, 1.2632e-02, 4.5088e-01,
                          7.3684e-02, 6.9035e-02, 3.0702e-02, 1.8246e-01, 8.5088e-03, 1.7719e-02,
                          3.3246e-02, 5.5702e-02, 1.8789e-01, 9.1228e-03, 4.7930e-01, 1.8404e-01,
                          4.2719e-02, 4.1228e-03, 2.4202e-01, 5.2632e-04, 2.1053e-03, 3.5965e-03,
                          7.1053e-03, 5.0439e-02, 1.6667e-03, 6.0219e-01, 0.0000e+00, 2.0939e-01,
                          8.7719e-05, 8.4211e-02, 8.7719e-05, 0.0000e+00, 8.7719e-05, 0.0000e+00,
                          4.7544e-02, 2.4211e-02, 5.6114e-01, 1.0526e-03, 7.0877e-02, 2.2807e-03,
                          1.5211e-01, 1.9298e-03, 1.0526e-03, 3.5088e-04, 1.1140e-02, 1.7368e-01,
                          2.1053e-03, 4.2105e-03, 8.2456e-03, 1.5175e-02, 2.1930e-02, 5.0614e-02,
                          1.7018e-02, 3.0702e-03, 4.0351e-03, 4.4956e-01, 1.5702e-02, 4.2982e-02,
                          1.6053e-02, 9.3860e-03, 2.7368e-02, 8.3658e-01, 1.0526e-03, 1.0526e-03,
                          1.9649e-02, 1.8158e-02, 7.3833e-01, 4.6140e-02, 3.7456e-02, 1.2456e-02,
                          5.5702e-02, 2.7105e-02, 7.6316e-03, 8.1237e-01, 7.1261e-02, 5.8006e-02,
                          5.6510e-02, 1.8528e-03, 1.6667e-03, 3.5000e-02, ])

print(pos_ratio)
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:02:35 2020

@author: cw817615
"""


"""
import pandas as pd

'''读取数据'''
io = r'C:/Users/dell/Desktop/regress_data2.xls'
col_name = ['性别', '年级', '时长', '使用频率', '使用数量', '专业类别', '是否为重度抑郁情绪']
data = pd.read_excel(io, sheet_name=0, names=col_name, header=None)


import numpy as np
#

X = data.iloc[:, 0:6].values     # 自变量
y = data.iloc[:, 6].values     # 因变量

##'''SMOTE的改进：Borderline-SMOTE处理过采样'''
# from imblearn.under_sampling import ClusterCentroids
# cc = ClusterCentroids(random_state=0)
# X_resampled, y_resampled = cc.fit_resample(X, y)

#from imblearn.over_sampling import RandomOverSampler

#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_sample(X, y)

from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# 合并数据
data_resampled = np.zeros([len(X_resampled[:, 0]), 7])
data_resampled[:, :6] = X_resampled
data_resampled[:, 6] = y_resampled

data_resampled2 = pd.DataFrame(data_resampled)
writer = pd.ExcelWriter(r'C:/Users/dell/Desktop/guocaiyang.xls')#创建数据存放路径
data_resampled2.to_excel(writer)
writer.save()#文件保存
writer.close()#文件关闭
"""


# with open(r"C:/Users/dell/Desktop/train.txt", encoding='utf-8') as f:
#     for item in f:
#         list.append(item)

# datalist = []
# for item in list:
#     x = item.split()
#     datalist.append(x)

# for item in datalist:
#     if item[12] == '1':
#         print(item)
