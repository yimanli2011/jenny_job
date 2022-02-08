
import os.path
import requests
import re
import urllib.parse

######  百度图片爬虫 ######
# 获取动态页面返回的文本
def get_page_html(page_url):

    headers = {
    'Host': 'image.baidu.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36',
    'Cookie': 'BAIDUID=0A4C55C2C07A4AFEA9A2962A4A32E9B6:FG=1; BIDUPSID=0A4C55C2C07A4AFEA9A2962A4A32E9B6; PSTM=1618885599; BDSFRCVID_BFESS=Mh-OJeC62lT9mbceq4wubZYHBetv3QnTH6ao3TWsR1SrLTPC1ZFtEG0P8U8g0Kubmgz-ogKKy2OTH9DF_2uxOjjg8UtVJeC6EG0Ptf8g0f5; H_BDCLCKID_SF_BFESS=tbCO_C0KJCI3qn7I55-_-tAObfo-etJyaR3uBT6vWJ5TMCo135rV-4t8KHOf-4jzymbG04bKtMj_ShPC-tP53h-y0GQgbf7nLgADhqQ23l02VKnae-t2ynLVhfJEe4RMW238oq7mWITUsxA45J7cM4IseboJLfT-0bc4KKJxbnLWeIJIjj6jK4JKjHutqTOP; H_PS_PSSID=33984_33820_31254_33848_33676_33607_34026; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; delPer=0; BAIDUID_BFESS=0A4C55C2C07A4AFEA9A2962A4A32E9B6:FG=1; PSINO=7; __yjs_duid=1_1c2df3da73c373af270afadce2b5f00e1621472126118; ab_sr=1.0.0_ZWEyNTQ3ODMwMWUyZmNmNTM5NWQxNDRlMDkyMWQyMGJhNjI3ZGZiZDkzNmEzM2MyOGUyNGFlOTE0YWEzYTliZjVlZTM1ZmJmMzhkOWQ4NDcwYTBlNjJmYjM3NDAxMTZl; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; BA_HECTOR=0l818h0101040gal0p1gabgv20r'
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
    url_real = re.findall('"thumbURL":"(.*?)",', text)
    return url_real


# 获取图片的content
def get_image_content(url_real):
    headers = {
        'Referer': url_real,
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
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
    root = 'D:/liyiman/datasets/crawl_imgs/tezhongcheliang/'
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
        url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord+=&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&word={}&z=&ic=0&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&step_word={}&pn={}&rn=30&gsm=1e&1541136876386='.format(
            keyword_quote, keyword_quote, i * 30)

        html = get_page_html(url)
        real_urls = parse_result(html)
        for real_url in real_urls:
            content = get_image_content(real_url)
            save_pic(real_url, content)


# 函数入口
if __name__ == '__main__':
    main()