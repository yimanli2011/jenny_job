import numpy as np
import onnxruntime
import cv2,os
from torchvision.transforms import transforms
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import os
import json


# def loader(img, size):
#     img = cv2.imread(img)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
#     img = np.array(img)
#     return img
#
#
# def preprocess(img):
#     transform_list = [
#         # transforms.ToTensor(),
#         transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
#     ]
#     transform = transforms.Compose(transform_list)
#
#     img = loader(img, (1000, 568))
#     img = Image.fromarray(img)
#     print(type(img),np.array(img).dtype)
#     img = transforms.ToTensor()(img)*255
#     print(img.dtype)
#     img = transform(img)
#     img = img.unsqueeze(0)
#     # print(img.shape)
#     #img = img.cuda()
#     return img
#
#
# imgpath = 'D:/liyiman/aa/json_img/'
# jsonpath = 'D:/liyiman/aa/json/'
# savepath = 'D:/liyiman/aa/save/'
#
#
# if __name__ == "__main__":
#     sess = onnxruntime.InferenceSession('D:/liyiman/models_nj/bcpmud/20210525_01/bcpmud_dynamic.onnx')
#     # sess = onnxruntime.InferenceSession('/data_01/data2/liyong/code/aiw/jishui.onnx')
#     input_name = sess.get_inputs()[0].name
#     # outputnames = ['score','label']
#     outputnames = ['output']
#
#     for img_f in os.listdir(imgpath):
#     # for impath in os.listdir('/data_01/data2/liyong/code/aiw/jishui_show/yuantu'):
#         # img_path = "/data_01/data2/liyong/code/aiw/jishui_show/yuantu/bb1eb11fc549947ab61d3a9a16bebf11.jpg"
#         img = cv2.imread(imgpath + img_f)
#         img_show = img.copy()
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         imgshow = img.copy()
#         """
#         img = cv2.resize(img, (512, 512),cv2.INTER_LINEAR)
#         # img = bin_linear(img, [512, 512])
#
#         img = np.float32(img)
#
#         for row in range(img.shape[0]):
#             for col in range(img.shape[1]):
#                 img[row,col] = (img[row,col] - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
#
#         img = img.swapaxes(1,2).swapaxes(0,1)
#         input = np.float32(np.expand_dims(img, axis=0))
#         """
#         img = np.array(preprocess(imgpath + img_f))
#
#         prediction = sess.run(outputnames, {input_name:img})
#         # prediction = sess.run(outputnames, {input_name:img})
#
#         # score = prediction[0]
#         label = prediction[0].astype(np.uint8)
#
#         line_image_show = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
#         for row in range(label.shape[0]):
#             for col in range(label.shape[1]):
#                 if label[:,row,col] == 0:
#                     line_image_show[row,col] = [0,0,0]
#                 elif label[:,row,col] == 1:
#                     line_image_show[row,col] = [255,0,0]
#                 elif label[:,row,col] == 2:
#                     line_image_show[row,col] = [0,255,0]
#                 else:
#                     line_image_show[row,col] = [0,0,255]
#         line_image_show = cv2.resize(line_image_show, (img_show.shape[1],img_show.shape[0]))
#         line_image_show = 0.5*line_image_show + 0.5*img_show
#         cv2.imwrite(savepath+img_f,line_image_show)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import os
import json
import os.path as osp

def PictureToMask(d_object, sourcePicture):
    '''得到原图的宽度和高度'''
    im = Image.open(sourcePicture)
    size = list(im.size)
    width = size[0]
    height = size[1]

    '''将图片的像素的宽度和高度换算成英寸的宽度和高度'''
    dpi = 80  # 分辨率
    ycwidth = width / dpi  # 宽度(英寸) = 像素宽度 / 分辨率
    ycheight = height / dpi  # 高度(英寸) = 像素高度 / 分辨率

    color = ["g", "r", "b", "y", "skyblue", "k", "m", "c"]
    fig, ax = plt.subplots(figsize=(ycwidth, ycheight))
    for region in d_object:
        '''将传进来的x轴坐标点和y轴坐标点转换成numpy数组，相加后转置成多行两列'''
        x = np.array(d_object[region][0])
        y = np.array(d_object[region][1]) * -1
        xy = np.vstack([x, y]).T
        '''
        #设置画框的背景图片为原图
        fig = plt.figure(figsize=(ycwidth,ycheight),dpi=dpi)
        bgimg = img.imread(sourcePicture)
        fig.figimage(bgimg)
        '''
        '''将numpy中的坐标连城线，绘制在plt上'''
        plt.plot(xy[:, 0], xy[:, 1], color=color[int(region)])
        plt.fill_between(xy[:, 0], xy[:, 1], facecolor=color[int(region)])  # 对该分割区域填充颜色
    plt.xticks([0, width])
    plt.yticks([0, -height])
    plt.axis("off")
    # 保存图片
    path = sourcePicture.rsplit(".", 1)[0]
    print(sourcePicture)
    print(path)
    plt.savefig(path + "-mask.png", format='png', bbox_inches='tight', transparent=True,
                dpi=100)  # bbox_inches='tight' 图片边界空白紧致, 背景透明
    # plt.show()


def getJson(filepath):
    '''从文件夹获取json文件内容，返回字典'''
    files = os.listdir(filepath)
    for file in files:
        if osp.splitext(file)[1] == ".json":
            jsonfile = filepath + file
            break
    jsonstr = open(jsonfile, "r", encoding="utf8").read()
    d_json = json.loads(jsonstr)
    # print(d_json)
    return d_json


def getPath():
    '''输入图片文件夹路径'''
    filepath = input("图片文件夹路径：")
    if filepath.endswith != "/" or filepath.endswith != "\\":
        filepath = filepath + "/"
    return filepath


def main():
    filepath = getPath()
    d_json = getJson(filepath)

    data = d_json.get('shapes')
    d_object = {}
    for obj in data:
        l_object = []
        for point in obj["points"]:
            l_object.append(point)
        d_object['region'] = l_object
    sourcePicture = filepath + os.path.basename(d_json["imagePath"])
    PictureToMask(d_object, sourcePicture)


if __name__ == "__main__":
    main()