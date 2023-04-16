""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from dataset import CIFAR100Train, CIFAR100Test, CIFAR100featureTrain, CIFAR100srTrain, CIFAR100featureTest

import horovod.torch as hvd

def get_network(args):
    """ return given network
    """

    if args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'resnet101_fpn':
        from models.resnetSR import resnet101_fpn
        net = resnet101_fpn()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net
    


#--------- dataloader for hr_resnet101, lr_resnet101
def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, type='hr'):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    
    """
    
    if type=='lr':
      transform_train = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize(16),
          transforms.Resize(32),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ])
    else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ])

    cifar100_training = CIFAR100Train('./data/cifar-100-python', transform=transform_train)
    #cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, type='hr'):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    
    if type=='lr':
      transform_test = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize(16),
          transforms.Resize(32),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ])
    else:
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ])
    cifar100_test = CIFAR100Test('./data/cifar-100-python', transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


#--------- dataloader for sr_cls
def get_training_feature_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([16,16]),
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_training=CIFAR100featureTrain('./data/cifar-100-python',featuere_path='./data/cifar100feature/train/hr/', transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        cifar100_training, num_replicas=hvd.size(), rank=hvd.rank())

    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, sampler=train_sampler)

    return cifar100_training_loader

def get_test_feature_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([16,16]),
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test=CIFAR100featureTest('./data/cifar-100-python',featuere_path='./data/cifar100feature/test/hr/', transform=transform_test)
                              
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader



# sr
def get_train_sr_dataloader(lr_path, hr_path, mean, std, batch_size=16, num_workers=2, shuffle=True):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = CIFAR100srTrain(lr_path, hr_path, transform)

    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_test_sr_dataloader(lr_path, hr_path, mean, std, batch_size=16, num_workers=2, shuffle=True):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = CIFAR100srTrain(lr_path, hr_path, transform)

    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]