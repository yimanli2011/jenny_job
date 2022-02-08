import cv2
import random
import numpy as np


def sp_noise(im, prob=0.005):
    '''prob: noise prob'''
    output = np.zeros(im.shape, np.uint8)
    thres = 1 - prob
    if random.random() > 0.4:
        return im
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = im[i][j]
    return output


def random_bright(im, delta=32):
    if random.random() < 0.4:
        delta = random.uniform(-delta, delta)
        im = im + delta
        im = im.clip(min=0, max=255)
    return im


def random_hue(im, delta=18.0):
    if random.random() < 0.4:
        im[:, :, 0] = im[:, :, 0] + random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] = im[:, :, 0][im[:, :, 0] > 360.0] - 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] = im[:, :, 0][im[:, :, 0] < 0.0] + 360.0

    return im


def random_saturation(im, lower=0.5, upper=1.5):
    if random.random() < 0.4:
        im[:, :, 1] = im[:, :, 1] * random.uniform(lower, upper)
        im = im.clip(min=0, max=255)
    return im


def random_contrast(im, lower=0.5, upper=1.5):
    if random.random() < 0.4:
        alpha = random.uniform(lower, upper)
        im = im * alpha
        im = im.clip(min=0, max=255)
    return im


def random_swap(im):
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    if random.random() < 0.3:
        swap = perms[random.randrange(0, len(perms))]
        im = im[:, :, swap]
    return im


def random_blur(im):
    if random.random() < 0.5:
        im = cv2.blur(im, (3, 3))
    return im


def build_transform(img):
    trans_lists = [random_bright, random_hue, random_saturation, random_contrast, random_swap, random_blur, sp_noise]
    for trans in trans_lists:
        img = trans(img)
    cv2.imwrite("res.jpg", img)
    return img


if __name__ == '__main__':
    im = cv2.imread("_4ytxohes.jpg")
    build_transform(im)
