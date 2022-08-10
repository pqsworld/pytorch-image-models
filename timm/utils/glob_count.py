""" JIT scripting/tracing utils

Hacked together by / Copyright 2020 Ross Wightman
"""

import os
from glob import glob
import matplotlib as plt
import cv2
from pathlib import Path

from pandas import set_option

path_count = r"/hdd/file-input/panq/dataset/lif/完整数据"


def find_sub_dirs(path, depth=2):
    path = Path(path)
    assert path.exists(), f'Path: {path} does not exist'
    depth_search = '*/' * depth
    search_pattern = os.path.join(path, depth_search)
    return list(glob(f'{search_pattern}'))


def list_bmp_directory(path, depth=1, type="*.bmp"):
    """
    递归取得文件夹下所有匹配路径
    Args:
        dir_imagefolder (_type_): 文件夹路径
        type (str, optional): 匹配字符串.   Defaults to "*.bmp".
    Returns````
        _type_: list
    """
    assert depth >= 0
    list_path = []
    for d in range(depth):
        list_path.extend(find_sub_dirs(path, d))
    print(*list_path, sep='\n')
    for p in range(len(list_path)):
        list_image_dir = []
        list_image_dir = glob(os.path.join(list_path[p], "**", type), recursive=True)
        print(list_path[p] + " : " + str(len(list_image_dir)))


def main():
    list_bmp_directory(path_count, 3, "*.bmp")


if __name__ == "__main__":
    main()
