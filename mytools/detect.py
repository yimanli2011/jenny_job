from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retina import Retina
from utils.box_utils import decode, decode_landm, multi_decode
import time
import torchvision
import os
print(torch.__version__, torchvision.__version__)

parser = argparse.ArgumentParser(description='RetinaPL')

parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.5, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.8, type=float, help='visualization_threshold')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def show_files(path, all_files):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            if cur_path.find(".jpg") > 0:
                all_files.append(cur_path)

    return all_files

def resize_img(img, size):

    img_mask = np.zeros(size).astype("uint8")
    h, w, _ = img.shape
    max_s = max(h, w)
    ratio = 640 / max_s
    img_r = cv2.resize(img, None, fx=ratio, fy=ratio)
    h_r, w_r, c = img_r.shape
    img_mask[:h_r, :w_r, :] = img_r

    return img_mask, ratio

def target_vertax_point(clockwise_point):
    #计算顶点的宽度(取最大宽度)
    w1 = np.linalg.norm(clockwise_point[0]-clockwise_point[1])
    w2 = np.linalg.norm(clockwise_point[2]-clockwise_point[3])
    w = w1 if w1 > w2 else w2
    #计算顶点的高度(取最大高度)
    h1 = np.linalg.norm(clockwise_point[1]-clockwise_point[2])
    h2 = np.linalg.norm(clockwise_point[3]-clockwise_point[0])
    h = h1 if h1 > h2 else h2
    #将宽和高转换为整数
    w = int(round(w))
    h = int(round(h))
    #计算变换后目标的顶点坐标
    top_left = [0,0]
    top_right = [w,0]
    bottom_right = [w,h]
    bottom_left = [0,h]
    return np.array([top_left,top_right,bottom_right,bottom_left],dtype=np.float32)

if __name__ == '__main__':
    
    torch.set_grad_enabled(False)
    cfg = cfg_mnet
    # net and model
    net = Retina(cfg=cfg, phase='test')
    net = load_model(net, './weights/best_mobilenet0.25_epoch_190_0.914.pth', True)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    resize = 1

    # testing begin
    contents = show_files("D:\data\warning_1", [])
    n = 1
    for img_name in contents:
        img_raw = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # img_raw = cv2.resize(img_raw, None, fx=resize, fy=resize)
        img, resize = resize_img(img_raw, (640, 640, 3))
        print(img_name)
        img = np.float32(img)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        rgb_mean = (103.53, 116.28, 123.675)  # bgr order
        rgb_std = (0.017429, 0.017507, 0.017125)
        img -= rgb_mean
        img *= rgb_std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print(loc.data.shape, conf.data.shape, landms.data.shape)
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        prior_data = prior_data[np.newaxis, :]
        boxes = multi_decode(loc.data, prior_data, cfg['variance'])
        # boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes.squeeze(0)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        landms = decode_landm(landms.data.squeeze(0), prior_data.squeeze(0), cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        print('priorBox time: {:.4f}'.format(time.time() - tic))
        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                print(text)
                b = list(map(int, b))

                # 定义对应的点
                points1 = np.float32([[b[5], b[6]], [b[7], b[8]], [b[9], b[10]], [b[11], b[12]]])
                points2 = target_vertax_point(points1)
                # 计算得到转换矩阵
                M = cv2.getPerspectiveTransform(points1, points2)
                # 实现透视变换转换
                processed = cv2.warpPerspective(img_raw, M, (int(points2[2][0]), int(points2[2][1])))
                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)

                cv2.circle(img_raw, (b[9], b[10]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (255, 0, 0), 4)
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                # 显示原图和处理后的图像
                # cv2.imshow("processed", processed)
                cv2.imwrite(os.path.join('D:\data\plate_test1', "res_%d.jpg"%n), processed)
                n += 1
            # img_raw = cv2.resize(img_raw, (512, 512))
            # cv2.imshow('image', img_raw)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()

