import timm
import torch
import numpy as np
import torchvision
from torchvision import transforms as transforms
import torchvision.transforms.functional as TF
import config
from collections import OrderedDict
import torch.nn as nn

# import pandas as pd
import time as ti
import argparse
import torch.nn.functional as F

# 显示所有列
# pd.set_option('display.max_columns',None)
# 显示所有行
# pd.set_option('display.max_rows',None)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


from torch.autograd import Variable


# from mobilenetv3_mcd import MNV3_large2,ImageClassifierHead
import timm
from mobilenetv3_small import MNV3_small_6_3_6_7 as MNV3_ben
from mobilenetv3_big import MNV3_small_6_3_6_7, ImageClassifierHead

# from MobileNet import MNV30811
from tqdm import tqdm
import datetime
import cv2
import itertools
import matplotlib.pyplot as plt

from shutil import copyfile

args = None
from shutil import move
from math import exp
import random
from utils.loss.triplet_loss import TripletLoss
from pathlib import Path

# 绘制混淆矩阵
class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, projection_dim, n_features):
        super(SimCLR, self).__init__()

        # self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        # self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.projector(x)
        return x


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)  # calmp夹断用法
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive, euclidean_distance


def random_str(slen=10):
    seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sa = []
    for i in range(slen):
        sa.append(random.choice(seed))
    return "".join(sa)


def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument(
        "-m",
        "--modelpath",
        # default="/home/panq/vendor/pytorch-image-models/output/train/20220708-151123-mobilenetv3_small_100-32/model_best.pth.tar",
        default="/home/panq/vendor/output/models_2022_07_04_07_mnv3-6159-compare-2/good_103.pth",
        help="train dir",
    )
    parser.add_argument(
        "-m2",
        "--modelpath2",
        default="/home/panq/vendor/pytorch-image-models/out/mobilevit/20220710-195704-mobilevit_s-32/model_best.pth.tar",
        help="train dir",
    )
    parser.add_argument(
        "-v",
        "--validdir",
        default="/hdd/file-input/panq/dataset/noid_6159_newmaterial_test",
        help="valid dir",
    )
    parser.add_argument("-t", "--thr", default=32768, type=int, help="valid dir")
    parser.add_argument("-d", "--df", default=False, type=bool, help="delflag")

    args = parser.parse_args()
    return args


class Dataset(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
    ):
        super(Dataset, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # sample = TF.erase(self.transform(sample),i=90,j=90,h=8,w=8,v=self.transform(sample).mean()+0.2)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class Dataset_ID(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
    ):
        super(Dataset_ID, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def __getitem__(self, index):
        path0, target = self.samples[index]
        path = Path(path0)
        sample = self.loader(path)
        # print(type(sample))
        if self.transform is not None:
            sample = self.transform(sample)
        if target == 0:
            temple_path = path.parent
        else:
            if str(path).find("valid") == -1:
                temple_path = Path(
                    str(path.parent.parent.parent.parent).replace("1neg", "0pos")
                )
                # print(temple_path)
                child_dir_list = [
                    child for child in temple_path.iterdir() if child.is_dir()
                ]
                temple_path = random.choice(child_dir_list)
                child_dir_list = [
                    child for child in temple_path.iterdir() if child.is_dir()
                ]
                temple_path = random.choice(child_dir_list)
                # print("temple_path:{}".format(temple_path))
            else:
                temple_path = Path(str(path.parent.parent).replace("1neg", "0pos"))
        # print("temple_path:{}".format(temple_path))
        temps = list(temple_path.rglob("*.bmp"))
        # print(len(temps))
        # if len(temps)<3:
        # print(len(temps))
        # random.shuffle(temps)
        temple_avg = torch.empty([sample.size(0), sample.size(1)])
        temp_num = 20
        temp_list = []
        for i in range(temp_num):
            # print("temple_path:{} temp:{}".format(temple_path,temp))
            # try:
            if str(temps[i % len(temps)]) != path0:
                temple = self.loader(temps[i % len(temps)])
            else:
                temple = self.loader(temps[temp_num % len(temps)])
                # continue
            # except:
            # print(temple_path)
            if self.transform is not None:
                temple = self.transform(temple)
            if temp_num == 0:
                break
            else:
                temp_list.append(temple)
            temp_num = temp_num - 1
        temple_avg = torch.stack(temp_list, dim=0)
        # print(temple_avg.shape)
        return sample, target, temple_avg, path0


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.xaxis.set_ticks_position("top")
    tick_marks = np.arange(len(classes))

    # plt.tick_params(labelsize=13)
    plt.xticks(tick_marks, classes, rotation=0, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize == True:
            string = "{:.2%}".format(cm[i][j])
        else:
            string = "{:}".format(cm[i][j])
        plt.text(
            j,
            i,
            string,
            fontsize=17,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def Otsu(img_mat):
    threshold, img_mat = cv2.threshold(
        img_mat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return img_mat


if __name__ == "__main__":
    softmax = nn.Softmax(dim=1)
    # ti.sleep(40000)
    # global args
    args = parse_args()
    softmax = nn.Softmax(dim=1)
    time = "{}_{}".format(
        datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M"), random_str(8)
    )
    # print(time)
    # imgsort = 1
    # logprint = 1
    # thr_mode = 1
    # solo_mode = 1
    # best_thr = 0.7398963730569952
    thr = args.thr
    print("threshold:{}".format(thr))
    delFlag = args.df
    # teacher_flag = 0
    softmax = nn.Softmax(dim=1)
    # copyEmptyDir = 0
    if delFlag:
        print("original picture has been moved!")
    device = torch.device("cuda")

    checkpoint = torch.load(args.modelpath)
    print("model path:{}".format(args.modelpath))

    # net0 = timm.create_model("mobilenetv3_small_100", num_classes=2).to("cuda")
    net0 = MNV3_ben(2).cuda()
    G = MNV3_small_6_3_6_7().cuda()
    C = ImageClassifierHead(2).cuda()
    # SC = SimCLR(projection_dim=2, n_features=48).cuda()
    # timm.models.helpers.load_checkpoint(net0, args.modelpath)
    net0.load_state_dict(net0, torch.load(args.modelpath, map_location="cuda:0")["net"])
    # G.load_state_dict(torch.load(args.modelpath2, map_location="cuda:0")["G"])
    # C.load_state_dict(torch.load(args.modelpath2, map_location="cuda:0")["C1"])
    # SC.load_state_dict(torch.load(args.modelpath,map_location='cuda:0')['SC'])
    """
    new_state_dict = OrderedDict()
    for k, v in checkpoint['G'].items():
        name = k[:] # remove `module.`
        #print(k)
        #print(v)
        new_state_dict[name] = v
    G=MNV3_large2().to(device)
    C=ImageClassifierHead(3).to(device)
    G.load_state_dict(new_state_dict)
    for k, v in checkpoint['C1'].items():
        name = k[:] # remove `module.`
        new_state_dict[name] = v
    C.load_state_dict(new_state_dict)
    """
    test_dir = args.validdir
    print("test dir:{}".format(test_dir))
    confusion_out_path = "./confusion_test%s" % time
    if not os.path.isdir(confusion_out_path):
        os.mkdir(confusion_out_path)
    # print(test_dir)
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    # test_set = torchvision.datasets.ImageFolder(test_dir, test_transform)
    test_set = Dataset_ID(test_dir, test_transform)
    print((test_set))
    classnum = len(test_set.classes)
    # print(test_set.imgs)
    appear_times = Variable(torch.zeros(classnum, 1))
    for label in test_set.targets:
        appear_times[label] += 1
    confusionmap = Variable(torch.zeros(classnum, classnum))
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=True, num_workers=8
    )  #  , pin_memory=True
    G.eval()
    C.eval()
    net0.eval()
    print("example")
    num = 0
    log = []
    dist_p_sum, dist_p_avg, dist_n_sum, dist_n_avg = 0, 0, 0, 0
    to_pil_image = transforms.ToPILImage()
    with torch.no_grad():
        if 1:
            with tqdm(total=len(val_loader), position=0, ncols=80) as pbar:
                (
                    total,
                    test_loss,
                    test_correct,
                    test_loss_ret,
                    correct,
                    size,
                    correct0,
                    size0,
                    correct1,
                    size1,
                ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                for batch_num, (data, target, temples, path) in enumerate(val_loader):
                    data, target, temples = data.cuda(), target.cuda(), temples.cuda()
                    _, out3 = net0(data)
                    # print(out3.shape)
                    p = softmax(out3)
                    p = p[:, 0]
                    pred_ben = [0 if i * 65536 > thr else 1 for i in p]
                    pred_ben = torch.tensor(pred_ben).cuda()
                    # print(p)
                    # pred_ben = out3.data.max(1)[1]
                    k = target.data.size()[0]
                    correct += pred_ben.eq(target.data).cpu().sum()
                    size += k
                    correct0 += (
                        torch.tensor(
                            [
                                1 if (i == j and i == 0) else 0
                                for (i, j) in zip(pred_ben, target)
                            ]
                        )
                        .cpu()
                        .sum()
                    )
                    size0 += -(target - 1).cpu().sum()
                    correct1 += (
                        torch.tensor(
                            [
                                1 if (i == j and i == 1) else 0
                                for (i, j) in zip(pred_ben, target)
                            ]
                        )
                        .cpu()
                        .sum()
                    )
                    size1 += target.cpu().sum()
                    pbar.update(1)
                print(
                    "v10  thr:{} total acc:{}/{}={:.2f}%,class0 err:{:.2f}%({}/{}),class1 err:{:.2f}%({}/{})".format(
                        thr,
                        correct,
                        size,
                        correct / size * 100,
                        (size0 - correct0) / size0 * 100,
                        (size0 - correct0),
                        size0,
                        (size1 - correct1) / size1 * 100,
                        (size1 - correct1),
                        size1,
                    )
                )
    for thr_dist in range(2, 9):  # [0.8657407407407406]:#(1,20):
        with tqdm(total=len(val_loader), position=0, ncols=80) as pbar:
            (
                total,
                test_loss,
                test_correct,
                test_loss_ret,
                correct,
                correct_id,
                size,
                correct0,
                correct0_id,
                size0,
                correct1,
                correct1_id,
                size1,
            ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            for batch_num, (data, target, temples, path) in enumerate(val_loader):
                data, target, temples = data.cuda(), target.cuda(), temples.cuda()
                out1 = C(G(data))
                # feat = feat.reshape(feat.size(0), -1)
                # out1 = C(feat)
                temples = temples.view(-1, 1, data.size(2), data.size(3))
                out2 = C(G(temples))
                out2 = out2.view(out1.size(0), -1, 2)
                # print(out2.shape)
                out1 = out1.view(out1.size(0), 1, 2).repeat(1, out2.size(1), 1)
                # print(out1.shape)
                out1 = out1.view(-1, 2)
                out2 = out2.view(-1, 2)
                dist = F.pairwise_distance(out1, out2)
                pred = [1 if i > thr_dist / 10 else (0 if i < 0.2 else 0) for i in dist]
                # pred = [1 if i>thr/10 else (-19 if i<0.1 else 0) for i in dist ]
                pred = torch.tensor(pred).cuda()
                pred = pred.view(-1, 20)
                pred = pred.sum(axis=1, keepdim=False)
                pred = [0 if i <= 10 else 1 for i in pred]
                pred = torch.tensor(pred).cuda()
                _, out3 = net0(data)
                p = softmax(out3)
                p = p[:, 0]
                pred_ben = [0 if i * 65536 > thr else 1 for i in p]
                pred_ben = torch.tensor(pred_ben).cuda()
                pred_jilian = [
                    0 if (i == j and i == 0) else 1 for (i, j) in zip(pred, pred_ben)
                ]
                pred_jilian = torch.tensor(pred_jilian).cuda()
                k = target.data.size()[0]
                correct += pred_jilian.eq(target.data).cpu().sum()
                correct_id += pred.eq(target.data).cpu().sum()
                size += k
                correct_list = [
                    1 if (i == j and i == 0) else 0
                    for (i, j) in zip(pred_jilian, target)
                ]
                correct0 += torch.tensor(correct_list).cpu().sum()
                correct0_id += (
                    torch.tensor(
                        [
                            1 if (i == j and i == 0) else 0
                            for (i, j) in zip(pred, target)
                        ]
                    )
                    .cpu()
                    .sum()
                )
                size0 += -(target - 1).cpu().sum()
                correct1 += (
                    torch.tensor(
                        [
                            1 if (i == j and i == 1) else 0
                            for (i, j) in zip(pred_jilian, target)
                        ]
                    )
                    .cpu()
                    .sum()
                )
                correct1_id += (
                    torch.tensor(
                        [
                            1 if (i == j and i == 1) else 0
                            for (i, j) in zip(pred, target)
                        ]
                    )
                    .cpu()
                    .sum()
                )
                size1 += target.cpu().sum()
                pbar.update(1)
            print(
                "v10+id  dist_thr:{} total acc:{}/{}={:.2f}%,class0 err:{:.2f}%({}/{}),class1 err:{:.2f}%({}/{})".format(
                    thr_dist / 10,
                    correct,
                    size,
                    correct / size * 100,
                    (size0 - correct0) / size0 * 100,
                    (size0 - correct0),
                    size0,
                    (size1 - correct1) / size1 * 100,
                    (size1 - correct1),
                    size1,
                )
            )
            print(
                "id  dist_thr:{} total acc:{}/{}={:.2f}%,class0 err:{:.2f}%({}/{}),class1 err:{:.2f}%({}/{})".format(
                    thr_dist / 10,
                    correct_id,
                    size,
                    correct_id / size * 100,
                    (size0 - correct0_id) / size0 * 100,
                    (size0 - correct0_id),
                    size0,
                    (size1 - correct1_id) / size1 * 100,
                    (size1 - correct1_id),
                    size1,
                )
            )
