#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import sys
import importlib
importlib.reload(sys)
import os
import os.path
import requests
import re

######  百度识图爬虫 ######

def get_page_html(page_url):

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
    url_real = re.findall('"thumbUrl":"(.*?)",', text)
    return url_real


# 获取图片的content
def get_image_content(url_real):
    headers = {
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
    root = 'D://baiduimage_mask//'
    path = root + url_real.split('/')[-1]+'.jpg'
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(path):
        path = path.replace('?', '')
        try:
            with open(path, 'wb') as f:
                f.write(content)
                print('图片{}保存成功，地址在{}'.format(url_real, path))
        except:
            pass
    else:
        pass


# 主函数
def main():
    print("="*50)
    depth = int(input("请输入要爬取的页数(每页30张图): "))
    for i in range(depth):

        url = 'https://graph.baidu.com/ajax/pcsimi?sign=1261da7e4aa6be9f0150c01643356632&srcp=&tn=pc&idctag=gz&sids=10005_10520_10968_10974_11031_17851_17070_18100_17201_17202_18311_19190_19162_19220_19218_19230_19268_19280_19550_19560_19660_19670_19807_20001_20020_20048_10000&logid=3772038564&gsid=&entrance=general&tpl_from=pc&pageFrom=graph_upload_pcshitu&page={}&'.format(i)

        html = get_page_html(url)
        real_urls = parse_result(html)
        for real_url in real_urls:
            real_url = real_url.replace('\\/', '/').replace('\\','//')
            content = get_image_content(real_url)
            save_pic(real_url, content)


# 函数入口
if __name__ == '__main__':
    main()
