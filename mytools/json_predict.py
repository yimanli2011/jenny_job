#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import os
import os.path as osp
import warnings
import copy

import numpy as np
import PIL.Image
from skimage import io
import yaml
from io import BytesIO
import argparse
from labelme import utils


NAME_LABEL_MAP = {
#下面的名字请以自身实际情况所更改
    '_background_': 0,
    "water": 1,
    # "zhantongdao": 2,
}

LABEL_NAME_MAP = {
#下面的名字请以自身实际情况所更改
    0: '_background_',
    1: "water",
    # 2: "zhantongdao",
}


def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


# similar function as skimage.color.label2rgb
def label2rgb(lbl, img=None, n_labels=None, alpha=0.5, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))

    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = PIL.Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz


def draw_label(label, img=None, label_names=None, colormap=None):
    import matplotlib.pyplot as plt
    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]

    if colormap is None:
        colormap = label_colormap(len(label_names))

    # print(label.shape)
    label_viz = label2rgb(label, img, n_labels=len(label_names))
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        if label_value not in label:
            continue
        if label_name.startswith('_'):
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append('{value}: {name}'
                          .format(value=label_value, name=label_name))
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    plt.switch_backend(backend_org)

    out_size = (label_viz.shape[1], label_viz.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default='F:/GPU/shuimian_json/shuimian/')
    parser.add_argument('-o', '--out', default='F:/微信/企业微信/WXWork Files/File/2021-07/jishui_out/')
    args = parser.parse_args()

    json_file = args.json_file

    list = os.listdir(json_file)
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])
        filename = list[i][:-5]       # .json
        if os.path.isfile(path):
            data = json.load(open(path))
            img = utils.image.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])  # labelme_shapes_to_label

            # modify labels according to NAME_LABEL_MAP
            lbl_tmp = copy.copy(lbl)
            for key_name in lbl_names:
                old_lbl_val = lbl_names[key_name]
                new_lbl_val = NAME_LABEL_MAP[key_name]
                lbl_tmp[lbl == old_lbl_val] = new_lbl_val
            lbl_names_tmp = {}
            for key_name in lbl_names:
                lbl_names_tmp[key_name] = NAME_LABEL_MAP[key_name]

            # Assign the new label to lbl and lbl_names dict
            lbl = np.array(lbl_tmp, dtype=np.int8)
            lbl_names = lbl_names_tmp

            output_dir = 'C:/Users/dell/Desktop/png/'
            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = draw_label(lbl, img, captions)
            out_dir = osp.basename(list[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(list[i]), out_dir)
            if not osp.exists(out_dir):
                os.mkdir(out_dir)

            PIL.Image.fromarray(img).save(osp.join(out_dir, '{}_gt.png'.format(filename)))
            PIL.Image.fromarray(lbl).save(osp.join(output_dir, '{}.png'.format(filename)))
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.png'.format(filename)))

            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
