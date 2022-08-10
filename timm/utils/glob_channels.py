""" JIT scripting/tracing utils

Hacked together by / Copyright 2020 Ross Wightman
"""

import os
import glob
import matplotlib as plt
import cv2

path_old = r"/hdd/file-input/panq/dataset/regray/old/"
path_new = r"/hdd/file-input/panq/dataset/regray/new/"
name_rep_src = "new"
name_rep_dst = "new_channeled_050"


def list_bmp_directory(dir_imagefolder, type="*.bmp"):
    """
    递归取得文件夹下所有匹配路径
    Args:
        dir_imagefolder (_type_): 文件夹路径
        type (str, optional): 匹配字符串.   Defaults to "*.bmp".
    Returns:
        _type_: list
    """
    list_image_dir = glob.glob(
        os.path.join(dir_imagefolder, "**", type), recursive=True
    )
    return list_image_dir


def weight_bmps(list_src_base, list_src_weight):
    """对两个路径list的图像进行基于权值的图像混合

    Args:
        list_src_1 (_type_):
        list_src_2 (_type_):
        name_folder_save (str, optional):     若保存，保存路径. Defaults to "".
        alpha (float, optional):     addWeighted的加权参数. Defaults to 0.5.
    """
    assert len(list_src_base) == len(list_src_weight)
    for i in range(len(list_src_base)):
        src_base = cv2.imread(list_src_base[i], cv2.IMREAD_GRAYSCALE)
        src_weight = cv2.imread(list_src_weight[i], cv2.IMREAD_GRAYSCALE)
        assert src_base.shape == src_weight.shape
        dst = cv2.merge([src_base, src_weight, src_base])

        if name_rep_dst != "":
            # use shape of new
            dir_save = list_src_weight[i].replace(name_rep_src, name_rep_dst)
            if not os.path.exists(dir_save):
                if not os.path.exists(os.path.dirname(dir_save)):
                    os.makedirs(os.path.dirname(dir_save))
                cv2.imwrite(dir_save, dst)
                print("Num  " + str(i))
            else:
                print("SKIP " + dir_save)
    print("works done! handle bmps {}".format(len(list_src_weight)))


def main():
    list_src_base = list_bmp_directory(path_old, "*.bmp")
    list_src_weight = list_bmp_directory(path_new, "*.bmp")
    weight_bmps(list_src_base, list_src_weight)


if __name__ == "__main__":
    main()
