import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from time import time

N_CHANNELS = 1
data_dir = "/hdd/file-input/panq/dataset/noid_6159/ht/train"

data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
data_set = torchvision.datasets.ImageFolder(data_dir, data_transform)

data_loader = torch.utils.data.DataLoader(data_set, shuffle=False, num_workers=20)

before = time()
mean = torch.zeros(1).cuda()
std = torch.zeros(1).cuda().cuda()
print("==> Computing mean and std..")
with tqdm(total=len(data_loader), position=0, ncols=80, desc=f"") as pbar:
    for batch_num, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        for i in range(N_CHANNELS):
            mean[i] += data[:, i, :, :].mean().cuda()
            std[i] += data[:, i, :, :].std().cuda()
        pbar.update(1)
mean.div_(len(data_set))
std.div_(len(data_set))
print(mean, std)

print("time elapsed: ", time() - before)
