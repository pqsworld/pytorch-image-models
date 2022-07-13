'''Fingerprint ASP training with PyTorch.'''
#%%
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,5"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import nni
from collections import OrderedDict
from torch.autograd  import  Function
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
#from LossF import TPCLoss
import torchvision
from torchvision import transforms as transforms
from torch import Tensor
from typing import Callable,Optional
#from resnet import resnet34
import numpy as np
import argparse
import time
#import matplotlib.pyplot as plt

from torch.autograd import Variable
from net0809 import ASP0809
from mobilenetv3_big import MNV3_small_6_3_6_7,ImageClassifierHead
import datetime 
import re
import utils
import random,string
from itertools import cycle
#from utils.analysis import collect_feature, tsne, a_distance
from utils.loss.center_loss_2c import CenterLoss
from utils.loss.triplet_loss import TripletLoss
from torch.utils.data import WeightedRandomSampler,DataLoader
from pathlib import Path
from shutil import copyfile
args = None
epoch_now = 0
class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, projection_dim, n_features):
        super(SimCLR, self).__init__()

        #self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        #self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x):
        #print(x.shape)
        x = self.projector(x)
        return x
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(torch.clamp(euclidean_distance - 0.1, min=0.0), 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        return loss_contrastive,euclidean_distance
class ContrastiveLoss_in(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.1):
        super(ContrastiveLoss_in, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(torch.clamp(euclidean_distance - self.margin, min=0.0), 2))     
        return loss_contrastive
class ContrastiveLoss_out(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.8):
        super(ContrastiveLoss_out, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        return loss_contrastive
def random_str(slen=10):
    seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sa = []
    for i in range(slen):
      sa.append(random.choice(seed))
    return ''.join(sa)
    #random_str(8)运行结果：l7VSbNEG
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)
def SoftCrossEntropy(inputs, target, reduction='sum'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target))/batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss
def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                             F.cross_entropy(outputs, labels) * (1. - alpha)
 
    return KD_loss

class MySoftCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduce = None, reduction: str = 'mean') -> None:
        super(MySoftCrossEntropy, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return SoftCrossEntropy(input, target, reduction='average')

def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument('-l','--lr', default=0.02, type=float, help='start learning rate')
    parser.add_argument('-e','--epoch', default=400, type=int, help='number of epochs')
    parser.add_argument('-b','--batch-size', default=256, type=int, help='batch size in each context')
    parser.add_argument('-c','--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda exist')
    parser.add_argument('-s','--traindir_s', default="/hdd/file-input/doodle/database/6159/王奔训练测试/整理/train_1", help='train dir')
    #parser.add_argument('-s','--traindir_s', default="/hdd/file-input/doodle/database/GC013add/valid", help='train dir')
    parser.add_argument('-t','--traindir_t', default="/hdd/file-input/doodle/database/GC013add/train", help='train dir')
    #parser.add_argument('-t','--traindir_t', default="/hdd/file-input/doodle/database/GC013add/valid", help='train dir')
    parser.add_argument('-v','--validdir', default="/hdd/file-input/doodle/database/6159/王奔训练测试/整理/valid_new", help='valid dir')
    parser.add_argument('-v2','--validdir2', default="/hdd/file-input/doodle/database/6159/王奔训练测试/整理/valid_new", help='valid dir2')
    parser.add_argument('-g','--gpu', default=5, type=int, help='gpu')
    parser.add_argument('-r','--resume',default=True,type=bool, help="resume from checkpoint")
    parser.add_argument('-n','--numclass',default=2,type=int, help="numclass")

    args = parser.parse_args()
    return args


class GRL(Function):
    def forward(self,input):
        return input
    def backward(self,grad_output):
        grad_input = grad_output.neg()
        return grad_input

Grl = GRL()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def log_print(log, content):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(time, content)
    log.append(time)
    log.append("  " + content + "\n")
    
class Dataset2(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None):
        super(Dataset2, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
    def __getitem__(self, index):
        path, target = self.samples[index]
        proj=0
        if path.find('N-BOE')!=-1:
            proj=0
        if path.find('N-WXN')!=-1:
            proj=1
        if path.find('R-BOE')!=-1:
            proj=2
        if path.find('R-WXN')!=-1:
            proj=3
        if path.find('EPROJ')!=-1:
            proj=4
        sample = self.loader(path)      
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,target,proj

class Dataset_SimCLR(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None):
        super(Dataset_SimCLR, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)      
        #print(sample.size)
        if self.transform is not None:
            sample1 = self.transform(sample)
        if self.target_transform is not None:
            sample2 = self.target_transform(sample)
        return sample1,target,sample2
class Dataset_ID(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None):
        super(Dataset_ID, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
    def __getitem__(self, index):
        path, target = self.samples[index]
        path = Path(path)
        sample = self.loader(path)      
        #print(path)
        if self.transform is not None:
            sample = self.transform(sample)
        #对应编号真手指找temple 作为模板
        if target == 0:
            temple_path = path.parent
        else:
            id = (str(path.parent.parent.parent.stem))
            temple_path = Path(str(path.parent.parent).replace("1neg","0pos").replace(id,"{:0>4d}".format((int(id)+epoch_now//10)%682+1)))
            #print(epoch_now)
        temps = list(temple_path.rglob('*.bmp'))
        try:
            temple = self.loader(random.choice(temps))
        except:
            print(temple_path)
        if self.transform is not None:
            temple = self.transform(temple)
        #当前文件夹采样temple2 
        temple_path2 = path.parent
        temps2 = list(temple_path2.rglob('*.bmp'))
        temple2 = self.loader(random.choice(temps2))
        if self.transform is not None:
            temple2 = self.transform(temple2)
        #异类内找temple3
        if target == 0:
            temple_path = Path(str(path.parent.parent.parent).replace("0pos","1neg"))
        else:
            temple_path = Path(str(path.parent.parent.parent.parent).replace("1neg","0pos"))
        child_dir_list = [child for child in temple_path.iterdir() if child.is_dir()]
        temple_path = random.choice(child_dir_list)
        child_dir_list = [child for child in temple_path.iterdir() if child.is_dir()]
        temple_path = random.choice(child_dir_list)
        temps = list(temple_path.rglob('*.bmp'))
        temple3 = self.loader(random.choice(temps))
        if self.transform is not None:
            temple3 = self.transform(temple3)
        return sample,target,temple,temple2,temple3
class Dataset_ID_valid(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None):
        super(Dataset_ID_valid, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
    def __getitem__(self, index):
        path, target = self.samples[index]
        path = Path(path)
        sample = self.loader(path)      
        #print(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if target == 0:
            temple_path = path.parent
        else:
            temple_path = Path(str(path.parent.parent).replace("1neg","0pos"))
        temps = list(temple_path.rglob('*.bmp'))
        temple = self.loader(random.choice(temps))
        if self.transform is not None:
            temple = self.transform(temple)
        return sample,target,temple

class Net(object):
    def __init__(self, args):
        self.best_thr = 0.5
        self.model = None
        self.opt_g = None
        self.opt_c1 = None
        self.opt_c2 = None
        self.opt_c3 = None
        self.opt_c_same = None
        self.opt_centerloss = None
        self.opt_SC = None
        self.lr = args.lr
        self.epoch = 0
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.criterion = None
        self.device = None
        self.cuda = torch.cuda.is_available()
        self.train_loader_s = None
        self.train_loader_t = None
        self.val_loader = None
        self.log = []
        self.traindir_s=args.traindir_s
        self.traindir_t=args.traindir_t
        self.validdir=args.validdir
        self.validdir2=args.validdir2
        self.resume_flag=args.resume
        self.num_k=4
        self.num_worker = 32   
        self.num_gpu = 2
        self.numclass = args.numclass
        self.ana = False
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
    def create_sampler(self, dataset, num_samples):
        classes_idx = dataset.class_to_idx
        appear_times = Variable(torch.zeros(len(classes_idx), 1))
        for label in dataset.targets:
            appear_times[label] += 1
        #appear_times[0]=appear_times[0]/7
        classes_weight = (1./(appear_times / len(dataset))).view( -1)
        print(classes_weight)
        print(len(dataset))
        weight=list(map(lambda x:classes_weight[x],dataset.targets))
        sampler = WeightedRandomSampler(weights=weight, num_samples = num_samples,replacement=True)
        return sampler
    def load_data(self):
        train_transform = transforms.Compose([#transforms.Resize([200, 180]),
                                              transforms.RandomRotation([-5,5]),
                                              transforms.Grayscale(),
                                              #transforms.RandomCrop([120,32]),
                                              #transforms.RandomResizedCrop([200, 180], scale=(0.9, 1.1), ratio=(1,1), interpolation=2),
                                              #transforms.ColorJitter(brightness=(0.8, 1.2), contrast=0, saturation=0, hue=0),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                              transforms.ToTensor(),
                                              #transforms.RandomErasing(p=0.1, scale=(0.01, 0.1), ratio=(0.3, 3.3)),
                                              #transforms.Normalize()
                                              ])
        train_transform2 = transforms.Compose([#transforms.Resize([200, 180]),
                                              transforms.RandomRotation([-180,180]),
                                              transforms.Grayscale(),
                                              #transforms.RandomCrop([180,160]),
                                              #transforms.Resize([200, 180]),
                                              transforms.RandomResizedCrop([200, 180], scale=(0.98, 1.02), interpolation=2),
                                              transforms.ColorJitter(brightness=(0.98, 1.02), contrast=0, saturation=0, hue=0),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                              #transforms.RandomErasing(p=0.01, scale=(0.01, 0.05), ratio=(0.3, 3.3)),
                                              transforms.ToTensor(),
                                              #transforms.RandomErasing(p=0.01, scale=(0.01, 0.05), ratio=(0.3, 3.3)),
                                              #transforms.Normalize()
                                              ])
        test_transform = transforms.Compose([#transforms.Resize([200, 180]),
                                             transforms.Grayscale(),
                                             #transforms.CenterCrop([120,32]),
                                             #.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),#transforms.ToTensor(),
                                             transforms.ToTensor(),
                                             #transforms.Normalize()
                                             ])   #(0.34347,), (0.0479,)   vivo数据集
        '''
        train_set_s = torchvision.datasets.ImageFolder(self.traindir_s, train_transform)
        train_s_sampler = self.create_sampler(train_set_s, self.batch_size)
        self.train_loader_s = torch.utils.data.DataLoader(train_set_s, sampler = train_s_sampler, batch_size=self.batch_size, shuffle=(train_s_sampler == None), num_workers=self.num_worker, drop_last=True, pin_memory=True) # 
        '''
        train_set_s = Dataset_ID(self.traindir_s, transform = train_transform)
        train_set_t = torchvision.datasets.ImageFolder(self.traindir_t, train_transform)
        num_samples = int(len(train_set_s))
        #num_samples = int(300000)
        #定义源域的train_loader
        train_sampler_s = self.create_sampler(train_set_s,num_samples)
        self.train_loader_s = DataLoader(train_set_s, sampler=train_sampler_s, batch_size=self.num_gpu*self.batch_size, shuffle=(train_sampler_s == None), num_workers=self.num_worker, drop_last=True, pin_memory=True) 
        #定义目标域的train_loader
        train_sampler_t = self.create_sampler(train_set_t,num_samples)
        self.train_loader_t = DataLoader(train_set_t, sampler=train_sampler_t, batch_size=self.num_gpu*self.batch_size, shuffle=(train_sampler_t == None), num_workers=self.num_worker, drop_last=True, pin_memory=True) # 
        test_set = Dataset_ID_valid(self.validdir, transform = test_transform)
        self.val_loader = torch.utils.data.DataLoader(test_set, batch_size=self.num_gpu*self.batch_size, shuffle=False, num_workers=self.num_worker, drop_last=False, pin_memory=True) # 
        test_set_s = torchvision.datasets.ImageFolder(self.validdir2, test_transform)
        self.val_loader_s = torch.utils.data.DataLoader(test_set_s, batch_size=self.num_gpu*self.batch_size, shuffle=False, num_workers=self.num_worker, drop_last=False, pin_memory=True) #         
    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum =0.9,nesterov=0.01):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum,nesterov=nesterov)
            #self.opt_g = nn.DataParallel(self.opt_g).cuda()
            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum,nesterov=nesterov)
            #self.opt_c1 = nn.DataParallel(self.opt_c1).cuda()
            #self.opt_c2 = optim.SGD(self.C2.parameters(),
            #                        lr=lr, weight_decay=0.0005,
            #                        momentum=momentum)
            #self.opt_c2 = nn.DataParallel(self.opt_c2).cuda()
            #self.opt_c3 = optim.SGD(self.C3.parameters(),
            #                        lr=lr, weight_decay=0.0005,
            #                        momentum=momentum)
            #self.opt_c3 = nn.DataParallel(self.opt_c3).cuda()
            #self.opt_c_same = optim.SGD(self.C_same.parameters(),
            #                        lr=lr, weight_decay=0.0005,
            #                        momentum=momentum)
            #self.opt_c_same = nn.DataParallel(self.opt_c_same).cuda()
            self.opt_centerloss = optim.SGD(self.centerloss.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum,nesterov=nesterov)
            self.opt_SC = optim.SGD(filter(lambda p: p.requires_grad, self.SC.parameters()),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum,nesterov=nesterov)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            #self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     #lr=lr, weight_decay=0.0005)
            #self.opt_c3 = optim.Adam(self.C3.parameters(),
            #                         lr=lr, weight_decay=0.0005)
            #self.opt_c_same = optim.Adam(self.C_same.parameters(),
            #                         lr=lr, weight_decay=0.0005)
            self.opt_centerloss = optim.Adam(self.centerloss.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_SC = optim.Adam(filter(lambda p: p.requires_grad, self.SC.parameters()),
                                    lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        #self.opt_c2.zero_grad()
        #self.opt_c3.zero_grad()
        #self.opt_c_same.zero_grad()
        self.opt_centerloss.zero_grad()
        self.opt_SC.zero_grad()
    def make_data_pairs(self,data: torch.Tensor, target: torch.Tensor):
        feat = self.G(data)
        #calculate same loss
        index = [i for i in range(len(feat))]
        random.shuffle(index)
        feat_seq = feat[:]
        feat_rand = feat_seq[index]
        feat_same = torch.cat((feat_seq,feat_rand),2)
        output_same = self.C_same(feat_same)
        target_seq = torch.Tensor([target[i] for i in range(len(target))])
        target_rand = target_seq[index]
        label_same = torch.Tensor([torch.tensor(1.) if target_seq[i]!=target_rand[i] else torch.tensor(0.) for i in range(len(target))]).to(self.device)
        return output_same,label_same,feat_seq,feat_rand
    def get_symbol(self):
        if self.cuda:
            torch.cuda.current_device()
            torch.cuda._initialized = True
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            log_print(self.log,"cuda")
        else:
            self.device = torch.device('cpu')
        self.G = MNV3_small_6_3_6_7().to(self.device)
        self.G = torch.nn.DataParallel(self.G).cuda()
        self.C1=ImageClassifierHead(self.numclass).to(self.device)
        self.C1 = torch.nn.DataParallel(self.C1).cuda()
        #self.C2=ImageClassifierHead(self.numclass).to(self.device)
        #self.C2 = torch.nn.DataParallel(self.C2).cuda()
        self.C3=ImageClassifierHead().to(self.device)
        #self.C3 = torch.nn.DataParallel(self.C3).cuda()
        #self.C_same=ImageClassifierHead(in_channel=96).to(self.device)
        #self.C_same=SameClassifierHead().to(self.device)
        self.SC=SimCLR(projection_dim=2, n_features=48).to(self.device)
        self.SC = torch.nn.DataParallel(self.SC).cuda()
        #self.C_same = torch.nn.DataParallel(self.C_same).cuda()
        self.criterion1 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.,0.,0.75]),reduction="mean",ignore_index=1).to(self.device)
        self.criterion2 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.,0.75,0.]),reduction="mean",ignore_index=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion3 = MySoftCrossEntropy().to(self.device)
        self.centerloss = CenterLoss().to(self.device)
        self.tripletloss = TripletLoss().to(self.device)
        self.ContrastiveLoss =ContrastiveLoss().to(self.device)
        self.ContrastiveLoss_in = ContrastiveLoss_in(margin=0.1).to(self.device)
        self.ContrastiveLoss_out = ContrastiveLoss_out().to(self.device)
        self.CosineEmbeddingLoss = nn.CosineEmbeddingLoss(reduction = "mean").to(self.device)
        self.softmax = nn.Softmax(dim=1)
        self.set_optimizer(which_opt='momentum', lr=self.lr)
        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(self.opt_g, mode = 'min',factor = 0.5, patience = 5, verbose = True)
        self.scheduler_c1 = optim.lr_scheduler.ReduceLROnPlateau(self.opt_c1, mode = 'min',factor = 0.5, patience = 5, verbose = True)
        #self.scheduler_c2 = optim.lr_scheduler.ReduceLROnPlateau(self.opt_c2, mode = 'min',factor = 0.5, patience = 5, verbose = True)
        #self.scheduler_c3 = optim.lr_scheduler.ReduceLROnPlateau(self.opt_c3, mode = 'min',factor = 0.5, patience = 5, verbose = True)
        #self.scheduler_c_same = optim.lr_scheduler.ReduceLROnPlateau(self.opt_c_same, mode = 'min',factor = 0.5, patience = 5, verbose = True)
        self.scheduler_centerloss = optim.lr_scheduler.ReduceLROnPlateau(self.opt_centerloss, mode = 'min',factor = 0.5, patience = 5, verbose = True)
        self.scheduler_SC = optim.lr_scheduler.ReduceLROnPlateau(self.opt_SC, mode = 'min',factor = 0.5, patience = 5, verbose = True)
    def train_normal(self):
        log_print(self.log,"\ntrain_normal fingerprint asp:")
        self.G.train()
        self.C1.train()
        #self.C2.train()
        #self.C3.train()
        #self.C_same.train()
        #self.centerloss.train()
        #self.SC.train()
        torch.cuda.manual_seed(1)
        total,train_loss,train_correct,train_loss_ret,proj_loss,proj_loss_ret,same_loss,same_loss_ret = 0,0,0,0,0,0,0,0        
        centerloss,centerloss_ret = 0,0
        teacher_loss,softtarget_loss,label_loss_077,tpc_loss,domain_proj_loss,db_len,loss_s = 0,0,0,0,0,len(self.train_loader_s),0
        thr,thr_all =0,0
        with tqdm(total=len(self.train_loader_s),position=0,ncols=80) as pbar:
            for  batch_num,(data, target,temples,temples2,temples3) in enumerate(self.train_loader_s):
                data, target, temples,temples2,temples3 = data.to(self.device), target.to(self.device), temples.to(self.device), temples2.to(self.device), temples3.to(self.device)
                p = float(batch_num + (self.epoch-1) * db_len ) / (self.epochs *db_len)
                alpha = torch.tensor(2. / (1. + np.exp(-10 * p)) - 1)
                self.reset_grad()
                feat = self.G(data)
                out1 = self.C1(feat)
                feat_temp =self.G(temples)
                out2 = self.C1(feat_temp)
                feat_temp2 =self.G(temples2)
                out3 = self.C1(feat_temp2)
                feat_temp3 =self.G(temples3)
                out4 = self.C1(feat_temp3)
                loss,dist = self.ContrastiveLoss(out1,out2,target)
                loss_in = self.ContrastiveLoss_in(out1,out3)
                loss_out = self.ContrastiveLoss_out(out1,out4)
                loss_total =loss+loss_in+0.1*loss_out
                loss_total.backward()
                self.opt_g.step()
                self.opt_c1.step()
                pred = [0 if i<self.best_thr else 1 for i in dist ]
                pred = torch.tensor(pred).to(self.device)
                train_correct += pred.eq(target.data).cpu().sum()
                train_loss += loss.detach().cpu().numpy()#loss.cpu().item()
                total += target.size(0)
                train_loss_ret = train_loss / (batch_num + 1)
                pbar.update(1)
        return train_loss_ret, train_correct / total

    def resume(self):
        #model_path=r'../项目兼容验证2.0/STANDALONE/model_STANDALONE_2021_06_23_10_QMHbOoe9/ckpt_144.pth'
        #model_path=r'./STANDALONE/model_STANDALONE_2021_07_06_02_Sd7mF9Q1/ckpt_112.pth'
        #model_path=r'./STANDALONE/model_STANDALONE_2021_07_15_11_zL7tqTCN/ckpt_138.pth'
        model_path=r'./STANDALONE/model_STANDALONE_2021_09_01_12_MuRWfRrl/ckpt_178.pth'
        #model_path=r'./STANDALONE/model_STANDALONE_2021_08_23_09_wcgWfrow/ckpt_138.pth'
        log_print(self.log,'Resuming from checkpoint...\n model_path:%s'%model_path)
        #checkpoint=torch.load(model_path, map_location={'cuda:6': 'cuda:0'})
        checkpoint=torch.load(model_path,map_location={'cuda:5': 'cuda:0'})
        '''
        #加载模型方式一:从预训练模型当中挑选指定层，依次复制
        dd = self.symbol.state_dict()
        ckpt_net = MNV3_large2(3).to(self.device)
        ckpt_net.load_state_dict(checkpoint['net'],strict=False)
        new_state_dict=ckpt_net.state_dict()
        for name,p in ckpt_net.named_parameters():
            if name in dd.keys() and name.startswith('features'):  
                dd[name]=new_state_dict[name]
        self.symbol.load_state_dict(dd)
        '''
        #加载模型方式二:调用load_state_dict方法完全复制
        #'''
        self.G.module.load_state_dict(checkpoint['G'])
        #self.C1.load_state_dict(checkpoint['C1'])
        #self.C2.load_state_dict(checkpoint['C2'])
        #'''
    def test(self):
        log_print(self.log,"test:")
        self.G.eval()
        self.C1.eval()
        #self.C2.eval()
        #self.C3.eval()
        #self.C_same.eval()
        self.centerloss.eval()
        self.SC.eval()
        #prob = np.empty(shape=[0,2])
        total,test_loss,test_correct,test_loss_ret,correct,size = 0,0,0,0,0,0
        with tqdm(total=len(self.val_loader),position=0,ncols=80) as pbar:
            with torch.no_grad():
                for  batch_num,(data, target,temples) in enumerate(self.val_loader):
                    data, target, temples = data.to(self.device), target.to(self.device), temples.to(self.device)
                    feat = self.G(data)
                    feat = feat.reshape(feat.size(0), -1)
                    out1 = self.C1(feat)
                    feat_temp =self.G(temples)
                    feat_temp = feat_temp.reshape(feat_temp.size(0), -1)
                    out2 = self.C1(feat_temp)
                    loss,dist = self.ContrastiveLoss(out1,out2,target)
                    pred = [0 if i<self.best_thr else 1 for i in dist ]
                    pred = torch.tensor(pred).to(self.device)
                    
                    test_loss += loss.detach().cpu().numpy()
                    k = target.data.size()[0]
                    correct += pred.eq(target.data).cpu().sum()
                    size += k
                    pbar.update(1)
                test_loss = test_loss / (batch_num+1)
        log_print(self.log, "test loss: %1.5f, test acc：%1.5f" % (test_loss, correct / size))
        return test_loss, correct / size
    
    def save(self,epoch):
        model_out_path = "%s/ckpt_%d.pth" % (model_path,epoch)     
        torch.save(self.model, model_out_path)
        log_print(self.log, "Checkpoint saved to {}".format(model_out_path))

    def start(self):
        global epoch_now
        log_print(self.log,"source train dir path:{}".format(self.traindir_s))
        log_print(self.log,"target train dir path:{}".format(self.traindir_t))
        log_print(self.log,"valid dir path:{}".format(self.validdir))
        log_print(self.log,"learning rate:{}    batch size:{}    gpu:{}".format(self.lr,self.batch_size,torch.cuda.current_device()))
        self.load_data()
        self.get_symbol()
        print_network(self.G)
        if 0:
            self.resume()
        train_accuracy = 0
        test_accuracy = 0
        train_result = [0., 0.]
        test_result = [0., 0.]
        self.test()
        for epoch in range(0, self.epochs + 1):
            self.epoch = epoch
            epoch_now = epoch
            train_result = self.train_normal()
            #'''
            t_SNE_term = 15
            #'''
            log_print(self.log, "Epoch[%03d]: %03d/%d    acc=%1.5f    lossvalue=%1.5f \tlearning_rate=%1.7f" % (epoch,epoch,self.epochs,train_result[1],train_result[0], get_lr(self.opt_g)))
            self.scheduler_g.step(train_result[0])
            self.scheduler_c1.step(train_result[0])
            #self.scheduler_c2.step(train_result[0])
            #self.scheduler_c3.step(train_result[0])
            #self.scheduler_c_same.step(train_result[0])
            self.scheduler_centerloss.step(train_result[0])
            self.scheduler_SC.step(train_result[0])
            if (train_result[1] > train_accuracy and train_result[1] > 0.90) or (train_result[1] > 0.92) or (epoch == self.epochs) or epoch ==1:
                state = {
                        'G': self.G.module.state_dict(),
                        'C1': self.C1.module.state_dict(),
                        'SC': self.SC.module.state_dict(),
                        #'C3': self.C3.module.state_dict(),
                        #'C_same': self.C_same.module.state_dict(),
                        'train_acc':train_result[1],
                        'test_acc': test_result[1],
                        'epoch': epoch,
                }
                self.model = state
                self.save(epoch)
                #self.analysis_model()
            train_accuracy = max(train_accuracy, train_result[1])
            test_term = 40
            if train_accuracy > 0.7 and epoch > 50:
                test_term = 1
            if (epoch%test_term == 0 and epoch > 50) or epoch ==1:
                test_result = self.test()
                #self.test_same()
                #当测试集准确率高于历史最高准确率时，保存模型，更新测试集历史最高准确率
                if test_result[1] > test_accuracy:
                    state = {
                        'G': self.G.module.state_dict(),
                        'C1': self.C1.module.state_dict(),
                        'SC': self.SC.module.state_dict(),
                        #'C_same': self.C_same.module.state_dict(),
                        'train_acc':train_result[1],
                        'test_acc': test_result[1],
                        'epoch': epoch,
                    }
                    self.model = state
                    self.save(epoch)
                    #self.analysis_model()
                test_accuracy = max(test_accuracy, test_result[1])
                log_print(self.log,'[%d]Accuracy-Highest=%1.5f  test_accuracy=%1.5f  lossvalue=%1.5f' % (epoch, test_accuracy,test_result[1],test_result[0]))
                #self.scheduler.step(test_result[0])
            f = open(model_path + "/log.txt", 'a')
            f.writelines(self.log)
            f.close()
            self.log = []
            
            if get_lr(self.opt_g) < 1e-5 and get_lr(self.opt_c1) < 1e-5:
                state = {
                    'G': self.G.module.state_dict(),
                    'C1': self.C1.module.state_dict(),
                    'SC': self.SC.module.state_dict(),
                    #'C_same': self.C_same.module.state_dict(),
                    'train_acc':train_result[1],
                    'test_acc': test_result[1],
                    'epoch': epoch,
                }
                self.model = state
                self.save(epoch)
                #self.analysis_model()
                break
        log_print(self.log,'the train work is over!')
def main():
    global model_path
    global device_ids
    global time
    #time.sleep(5400)
    #print("wait 5400s")
    #device_ids = [4,5,6,7]
    args = parse_args()
    experiment_id=nni.get_experiment_id()
    trial_id=nni.get_trial_id()
    time = datetime.datetime.now().strftime('_%Y_%m_%d_%H')
    model_path = "./%s/model_%s%s_%s"%(experiment_id,trial_id,time,random_str(8))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    copyfile(__file__, Path(model_path)/"train.py") # Copies the file
    #SEED = 0
    #np.random.seed(SEED)
    #torch.manual_seed(SEED)
    #torch.cuda.manual_seed_all(SEED)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    #torch.cuda.set_device(args.gpu)
    net = Net(args)
    net.start()


if __name__ == '__main__':
    main()
    
