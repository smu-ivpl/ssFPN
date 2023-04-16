""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform


    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        filename = self.data['filenames'.encode()][index]
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return image, label #, filename.decode('utf8')

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return image, label

class CIFAR100featureTrain(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, featuere_path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

        # hr feature
        filenames = self.data['filenames'.encode()]
        label = self.data['fine_labels'.encode()]

        self.sr_path=[]
        for i in range(len(self.data['fine_labels'.encode()])):
            npy_path = os.path.join(featuere_path,str(label[i]),filenames[i].decode('utf8'))
            npy_path = npy_path.replace('.png','.npy')
            self.sr_path.append(npy_path)


    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))
        

        # hr feature
        hr_feature  = torch.Tensor(np.load(self.sr_path[index], allow_pickle=True))


        if self.transform:
            image = self.transform(image)
        return image, label, hr_feature

class CIFAR100featureTest(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, featuere_path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

        # hr feature
        filenames = self.data['filenames'.encode()]
        label = self.data['fine_labels'.encode()]

        self.sr_path=[]
        for i in range(len(self.data['fine_labels'.encode()])):
            npy_path = os.path.join(featuere_path,str(label[i]),filenames[i].decode('utf8'))
            npy_path = npy_path.replace('.png','.npy')
            self.sr_path.append(npy_path)


    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))
        

        # hr feature
        hr_feature  = torch.Tensor(np.load(self.sr_path[index], allow_pickle=True))


        if self.transform:
            image = self.transform(image)
        return image, label, hr_feature


class CIFAR100srTrain(Dataset):
    def __init__(self, lr_path, hr_path, transform=None):
        
        self.label_list = sorted(os.listdir(lr_path))
        self.lr_path=[]
        for dirpath, _, fnames in sorted(os.walk(lr_path)):
            for fname in sorted(fnames):
                lr_path = os.path.join(dirpath, fname)
                self.lr_path.append(lr_path)
        
        self.transform = transform

        # hr feature
        self.hr_path=[]
        for dirpath, _, fnames in sorted(os.walk(hr_path)):
            for fname in sorted(fnames):
                hr_path = os.path.join(dirpath, fname)
                self.hr_path.append(hr_path)

    def __len__(self):
        return len(self.lr_path)

    def __getitem__(self, index):

        lr_image = Image.open(self.lr_path[index]).convert("RGB")
        hr_image = Image.open(self.hr_path[index]).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

