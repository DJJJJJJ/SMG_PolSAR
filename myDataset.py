import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.io


def label2int(label):
    label_name = {'B200': 0, 'GXR1': 1, 'H500': 2, 'JX4D': 3, 'JX493': 4, 'PRA1': 5, 'S90': 6, 'T5G340': 7,
                  'V5': 8, 'W306': 9}
    return label_name[label]

def int2label(label):
    label_name = {0: 'B200', 1: 'GXR1', 2: 'H500', 3: 'JX4D', 4: 'JX493', 5: 'PRA1', 6: 'S90', 7: 'T5G340',
                  8: 'V5', 9: 'W306'}
    return label_name[label]

class PolSARImageDataset_zz(Dataset):
    def __init__(self, txt_file, filetype, transform=None):
        self.data = []
        self.labels = []
        self.label_to_idx = {}
        self.transform = transform
        self.filetype = filetype
        self.rootpath = 'F:\\data\\0603data\\'
        # 读取 txt 文件
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split()
                # if label not in self.label_to_idx:
                #     self.label_to_idx[label] = len(self.label_to_idx)  # 为每个类别分配唯一索引
                self.data.append(self.rootpath + self.filetype+'\\' + path)
                # self.labels.append(self.label_to_idx[label])
                self.labels.append(label2int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = torch.tensor(self.labels[idx])
        # 加载.mat 文件
        if self.filetype == 'T9':
            image = scipy.io.loadmat(img_path)['T9']
        elif self.filetype == 'S8':
            image = scipy.io.loadmat(img_path)['S8']
        elif self.filetype == 'PSCP' or self.filetype == 'PSCP_T9':
            image = scipy.io.loadmat(img_path)['PSCP']
        elif self.filetype == 'PauliRGB':
            # 将.mat后缀替换为.bmp
            img_path = img_path.replace('.mat', '.png')
            # 加载rgb图像
            image = Image.open(img_path)
        elif self.filetype == 'PCF':
            image = scipy.io.loadmat(img_path)['PCF']
        elif self.filetype == 'dpcp':
            R1 = scipy.io.loadmat(img_path)['R1']
            R2 = scipy.io.loadmat(img_path)['R2']
            R3 = scipy.io.loadmat(img_path)['R3']
            R4 = scipy.io.loadmat(img_path)['R4']
            # 沿着第三个维度拼接，并且R1 R2 R3 R4的维度不一样
            image = np.concatenate((R1, R2, R3, R4), axis=2)
        else:
            # 默认图片
            image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def get_label_to_idx(self):
        return self.label_to_idx
class PolSARImageDataset_Gotcha(Dataset):
    def __init__(self, txt_file, filetype, transform=None):
        self.data = []
        self.labels = []
        self.label_to_idx = {}
        self.transform = transform
        self.filetype = filetype
        self.rootpath = 'F:\\data\\PolSARVehicleDataSet\\Gotcha-Dataset\\'
        # 读取 txt 文件
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split()
                # if label not in self.label_to_idx:
                #     self.label_to_idx[label] = len(self.label_to_idx)  # 为每个类别分配唯一索引
                self.data.append(self.rootpath + self.filetype+'\\' + path)
                # label由str转为int
                self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = torch.tensor(self.labels[idx])
        # 加载.mat 文件
        if self.filetype == 'T9':
            image = scipy.io.loadmat(img_path)['T9']
        elif self.filetype == 'S8':
            image = scipy.io.loadmat(img_path)['S8']
        elif self.filetype == 'PSCP' or self.filetype == 'PSCP_T9':
            image = scipy.io.loadmat(img_path)['PSCP']
        elif self.filetype == 'PauliRGB':
            # 将.mat后缀替换为.bmp
            img_path = img_path.replace('.mat', '.png')
            # 加载rgb图像
            image = Image.open(img_path)
        elif self.filetype == 'PCF':
            image = scipy.io.loadmat(img_path)['PCF']
        elif self.filetype == 'dpcp':
            R1 = scipy.io.loadmat(img_path)['R1']
            R2 = scipy.io.loadmat(img_path)['R2']
            R3 = scipy.io.loadmat(img_path)['R3']
            R4 = scipy.io.loadmat(img_path)['R4']
            # 沿着第三个维度拼接，并且R1 R2 R3 R4的维度不一样
            image = np.concatenate((R1, R2, R3, R4), axis=2)
        else:
            # 默认图片
            image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

