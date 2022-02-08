import os
import shutil

# 将文件夹A里在txt文档中的图片移至文件夹B
path_A = 'D:/liyiman/datasets/Face_attr_mask/copy_mask_test_1_imgs/'
# path_A = 'D:/liyiman/datasets/JDWG/VOC_JDWG_Det/JPEGImages/'
path_B = 'D:/liyiman/datasets/ZNV_data/driver_attr_test/dd/'
path_txt = [
            # 'D:/liyiman/datasets/ZNV_data/ZTC_chepai_Det_polygon_01/VOC2007/ImageSets/Main/trainval.txt',
            # 'D:/liyiman/datasets/ZNV_data/ZTC_chepai_Det_polygon_02/VOC2007/ImageSets/Main/trainval.txt',
            'D:/liyiman/datasets/Face_attr_mask/mask_test_1_list.txt',
            # 'D:/liyiman/datasets/ZNV_data/driver_attr_03/val_1104_new.txt',
            # 'D:/liyiman/datasets/ZNV_data/driver_attr_05/val_1104_new.txt',
            # 'D:/liyiman/datasets/JDWG/VOC_JDWG_Det/ImageSets/Main/test.txt'
           ]


list_A = os.listdir(path_A) # 单层目录
os.makedirs(path_B, exist_ok=True)
result = []
for file in path_txt:
    with open(file, 'r', encoding="utf-8") as f:
        data = f.read().split('\n')
        result += data

res = [os.path.splitext(item.split(' ')[0].split('/')[-1])[0] for item in result]
for img in list_A:
    # img_name = os.path.splitext(img)[0]
    # if img_name in result:
    if os.path.splitext(img)[0] in res:
        old_path = os.path.join(path_A, img)
        new_path = os.path.join(path_B, img)
        shutil.move(old_path, new_path)


# 处理双层目录
def get_imglist(src, img_list):
    for file in os.listdir(src):
        cur_path = os.path.join(src,file)
        if os.path.isdir(cur_path):
            get_imglist(cur_path,img_list)
        else:
            img_list.append(cur_path)
    return img_list

# 找出存在于txt的图片列表中而未存在于A文件夹的图片列表
# list_A = []
# list_A = get_imglist(path_A, list_A)
# list_A = [item.replace('D:/liyiman/datasets/Face_attr_mask/copy_mask_test_1_imgs/', '') for item in list_A]
# list_A = [item.replace('\\', '/') for item in list_A]
# out = []
# result = []
# for file in path_txt:
#     with open(file, 'r', encoding="utf-8") as f:
#         data = f.read().split('\n')
#         result += data
#
# for res in result:
#     if res not in list_A:
#         out.append(res)
#
# print(out)