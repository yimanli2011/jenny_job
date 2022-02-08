import pickle
import os
from tqdm import tqdm


def filter(input, output, eta, file):
    paths = [x.strip() for x in open(file).readlines()]
    f = open(output, 'w')
    input = pickle.load(open(input, 'rb'))
    gender = input['gender']
    for i, s in enumerate(tqdm(gender)):
        s = s[0]
        if max(s) / min(s) < eta:
            f.write(paths[i] + '\n')


if __name__ == '__main__':
    filter('/data_01/data2/liyong/aiw/face/output.pkl', '/data_01/data2/liyong/aiw/face/output.txt',
           10, '/data_01/data2/puheng/FaceAttribute/new_test0.txt')