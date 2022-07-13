 # -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:35:50 2021

@author: suanfa
"""
import os
import shutil
import re
from shutil import move
from shutil import copyfile
import argparse
import random
from pathlib import Path
import math
def parse_args():
    parser = argparse.ArgumentParser(description="divide database")
    parser.add_argument('-p',"--path",default=r"\\?\Z:\doodle\database\6159\train\1notfp\防伪部分", help = "DB path")
    parser.add_argument("--r",default=0.0, help = "division ratio",type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    ratio = args.r
    fpdir = Path(path).parent/"整理"/"0pos"
    notfpdir = Path(path).parent/"整理"/"1neg"
    fpdir.mkdir(parents = True,exist_ok =True)
    notfpdir.mkdir(parents = True,exist_ok =True)
    list_fp = []
    list_notfp = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            basePath=os.path.join(root,d)
            if basePath.find('0pos')!=-1 or basePath.find('1neg')!=-1:
                continue
            list1=os.listdir(basePath)
            if len(list(Path(basePath).rglob('*.bmp')))==0:
                continue
            for sub in list1:
                subPath=os.path.join(basePath,sub)
                if os.path.isdir(subPath):#不是最深层文件夹 跳过
                    break
                elif subPath.find('.fmi')==-1 and subPath.find('.bmp')==-1:#删除非图像文件:
                    os.remove(subPath)
                    continue
                else:
                    if subPath.find("0fp")==-1:
                        list_notfp.append(basePath)
                    else:
                        list_fp.append(basePath)
                    break
    print(len(set(list_fp)))
    print(len(list_notfp))
    random.shuffle(list_fp)
    random.shuffle(list_notfp)
    temple_num = math.ceil(len(list_fp)/5)
    classes = 1
    #i = 0
    for i in range(len(list_fp)):
        src = list_fp[i%len(list_fp)]
        if len(list(Path(src).rglob("*.bmp")))>0:
            shutil.copytree(src,Path(fpdir)/'{:0>4d}'.format(i//5+1)/'L{:d}'.format(i%5))
            #i = i+1
    if path.find("valid")!=-1:
        len_notfp=len(list_notfp)
    else:
        len_notfp=len(list_notfp)
    #i = 0
    for i in range(len(list_fp)):
        #print(str(Path(list_notfp[i%len(list_notfp)]).stem))
        #if str(Path(list_notfp[i%len(list_notfp)]).stem).find("Base") != -1:
            #print(list_notfp[i%len(list_notfp)])
        src = list_notfp[i%len(list_notfp)]
        if len(list(Path(src).rglob('*.bmp')))>0:
            shutil.copytree(src,Path(notfpdir)/'{:0>4d}'.format(i//(5*classes)+1)/'L{:d}'.format(i%(5*classes)//classes)/'C{:0>3d}'.format(i%(5*classes)%classes))
            #shutil.copytree(src,Path(notfpdir)/'{:0>4d}'.format(i//5+1)/'L{:d}'.format(i%5))
            #i = i+1
        #else:
            #print("{}:{}".format(src,len(list(Path(src).rglob('*.bmp')))))
        