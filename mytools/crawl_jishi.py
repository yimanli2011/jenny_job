
import os.path
import requests
import re
import urllib.parse

######  百度图片爬虫 ######
# 获取动态页面返回的文本
def get_page_html(page_url):

    headers = {
    'Host': 'www.vcg.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
    'Cookie':'acw_tc=2760827416373016053124806ea1a4d492f4b09bd08ee64e5c5794134f7fb3; uuid=72d912dd-fd33-4cb9-af17-22a705c7b9c7; clientIp=123.138.236.90; sajssdk_2015_cross_new_user=1; Hm_lvt_5fd2e010217c332a79f6f3c527df12e9=1637301587; _uab_collina=163730158718752535183978; _ga=GA1.2.1848844299.1637301588; _gid=GA1.2.341910665.1637301588; fingerprint=58396687600fdcfc8f9e4425668c9d86; name=18729225664; api_token=ST-878-9bffa7bdb43609195343188f137d69700; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2288cb7d3304a768a95e22ef93ea5e89805%22%2C%22first_id%22%3A%2217d36c7823e245-04fece585a75da-3e604809-1296000-17d36c7823f2%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22%24device_id%22%3A%2217d36c7823e245-04fece585a75da-3e604809-1296000-17d36c7823f2%22%7D; connect.sid=s%3ALCTk5LE4ZiXh3tooboYiLRGogCehF0pZ.aczvEGERC%2B0yYWkbIHn%2FrtbYcfBTy%2BImNobgaxPz2uw; Hm_lpvt_5fd2e010217c332a79f6f3c527df12e9=1637301646; _fp_=eyJpcCI6IjEyMy4xMzguMjM2LjkwIiwiZnAiOiI1ODM5NjY4NzYwMGZkY2ZjOGY5ZTQ0MjU2NjhjOWQ4NiIsImhzIjoiJDJhJDA4JG5PaWtvNW41MTh6VjNhSUp3QjRGb3V6alhMQ1FWOTZNcnA4ZHRsTDJJMlMva3FZWkVIMEp5In0%3D; ssxmod_itna=eqAxRQ0=itnDumDzgmqG=25DKQxWuKxhc0DDtFDl=loxA5D8D6DQeGTbuevkbKkQeSW4ahz7OnKad9jeR+Pfblgmfp/i4B3DEx065Kmm4iiyDCeDIDWeDiDGR=DFxYoDervCMuD3qGrDlKDRx07l65DWcF4CnI1CoYse4D1N7vo6xG1DQ5DsZrvknKD0goAkDi3fCKGD40OD09hngPhDBRNeA+xmBwxElG3ej2qem0P3e+eHCCaQT04bWQxHGD4vBRDFjZKDDcD1jjn4D===; ssxmod_itna2=eqAxRQ0=itnDumDzgmqG=25DKQxWuKxhc0DDtD6p+ZD0yuH403KsYKcHdsrUgkIx2cDhozYCqrhd4t7wRGgK70aimoG=wIoLuYYlRR++1Iu95XD8EBEAu7HkuLGfxdtvSN45lq/zlGMo/lT=84n0OtOoZ9f0ZFQXM7ejPrb6dIrj++LdtuCTB08=mbLUEATW=HLYf7=EHHjUFsT=qF7x=HAXLIOxXRr9Oi8A+9dLGRW70RfY8/cO5vf6p7zj+/=v35dAjP4bXAOrwP3a014MpRPgODXA6TK9zP9OXbH3sx9rlENsluRFc0s4os2o7U0tsEGvOh4eGYo5V3G6fN1brYC3=m+7mGbjaPlwDQwdb=inr2oYmmhmlwjmGUQO9kGPgxeiINGHelxagwP7oxxeVld=kKxhAKaA5PFPg3ZhEeSEhRoXBECYPD7QTC+d0uUpLkmo6Ra+G6X8G0Uh=OI5C+8lh9FutnjFOD9S99z4ktAYnFkBG+7GuIFp7+hpG//hV/pTDGcDG7rhDOAxeOxHiDD='
    }
    try:
        r = requests.get(page_url, headers=headers)
        if r.status_code == 200:
            # print(r.content.decode('utf-8'))
            r.encoding = r.apparent_encoding
            return r.text
        else:
            print('请求失败')
    except Exception as e:
        print(e)


# 从文本中提取出真实图片地址
def parse_result(text):
    url_real = re.findall('"equalw_url":"(.*?)",', text)
    return url_real


# 获取图片的content
def get_image_content(url_real):
    headers = {
        'Referer': 'https://www.vcg.com/creative-image/shuimianpingziwuran/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
    }

    try:
        r = requests.get('http:'+url_real, headers=headers)
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
    path = root + url_real.split('/')[-1]+'.jpg'
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(path):
        path = path.replace('?', '')
        with open(path, 'wb') as f:
            f.write(content)
            print('图片{}保存成功，地址在{}'.format(url_real, path))
    else:
        pass


# 主函数
def main():
    print("="*50)
    keyword = input('请输入你要查询的关键字: ')
    keyword_quote = urllib.parse.quote(keyword)
    depth = int(input("请输入要爬取的页数(每页30张图): "))
    for i in range(depth):
        url = 'https://www.vcg.com/api/common/searchAllImage?page={}&phrase=%E6%B0%B4%E9%9D%A2%E7%93%B6%E5%AD%90%E6%B1%A1%E6%9F%93&transform={}&uuid=2FI10H_e3a291ffea49700e40b86efa4ecca00e&productId=100003'.format(
            i * 30,keyword_quote)

        html = get_page_html(url)
        real_urls = parse_result(html)
        for real_url in real_urls:
            content = get_image_content(real_url)
            save_pic(real_url, content)


# 函数入口
if __name__ == '__main__':
    main()