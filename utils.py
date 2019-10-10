import torchvision
import torch
import os
import pickle
import sys
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import json
import matplotlib.pyplot as plt
import seaborn as sns


def get_network(args, num_classes, use_gpu=True):
    """ return given network
    """
    if args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=num_classes)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.cuda()

    return net


def get_train_loader(dataset='mnist', accepted_class_labels=[], norm_lambda=1.0, batch_size=64, num_workers=4, pin_memory=True):
    train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    if dataset.lower() == 'mnist':
        train_dataset = MNISTSubDataset(root='./data', train=True, download=True, transform=train_transform, include_list=accepted_class_labels, norm_lambda=norm_lambda)
    elif dataset.lower() == 'cifar100':
        train_dataset = CIFAR100SubDataset(root='./data', train=True, download=True, transform=train_transform, include_list=accepted_class_labels, norm_lambda=norm_lambda)
    else:
        raise AssertionError('Dataset {} is currently not supported'.format(dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader


def get_test_loader(dataset='mnist', accepted_class_labels=[], batch_size=64, num_workers=4, pin_memory=True):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if dataset.lower() == 'mnist':
        train_dataset = MNISTSubDataset(root='./data', train=False, download=True, transform=test_transform
                                       , include_list=accepted_class_labels)
    elif dataset.lower() == 'cifar100':
        train_dataset = CIFAR100SubDataset(root='./data', train=False, download=True, transform=test_transform
                                          , include_list=accepted_class_labels)
    else:
        raise AssertionError('Dataset {} is currently not supported'.format(dataset))
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size
                                              , num_workers=num_workers, pin_memory=pin_memory)
    return test_loader


class MNISTSubDataset(torchvision.datasets.MNIST):
    def __init__(self, *args, include_list=[], norm_lambda=1.0, **kwargs):
        super(MNISTSubDataset, self).__init__(*args, **kwargs)
        self.norm_lambda = norm_lambda

        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = (labels.reshape(-1, 1) == include).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        y = self.targets[index]
        x = self.data[index]
        random_index = np.random.randint(0, len(self.targets), 1)
        random_x = self.data[random_index[0]]
        random_y = self.targets[random_index[0]]
        random_scale = np.random.uniform(-self.norm_lambda, self.norm_lambda, 1)

        return x, y, random_scale, random_x, random_y #I1, Y1, A, I2, Y2

class CIFAR100SubDataset(torchvision.datasets.CIFAR100):
    def __init__(self, *args, include_list=[], **kwargs):
        super(CIFAR100SubDataset, self).__init__(*args, **kwargs)
        self.norm_lambda = kwargs.get('norm_lambda', 1.0)

        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = (labels.reshape(-1, 1) == include).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        y = self.targets[index]
        x = self.data[index]
        random_index = np.random.randint(0, len(self.targets), 1)
        random_x = self.data[random_index]
        random_y = self.targets[random_index]
        random_scale = np.random.uniform(-self.norm, self.norm, 1)

        return x, y, random_scale*x, random_scale*y, x+random_x, (y, random_y) #I1, Y1, L*I1, L*Y1, (I1+I2), (Y1+Y2)


def save_setting(setting, path):
    with open(path+'/setting.json', 'w') as f:
        json.dump(setting, f, indent=4)


def plot_norm_losses(l_a, l_t, l_z, path):
    xticks = ('Alpha', 'Triangle', 'ZERO')
    y = [l_a, l_t, l_z]
    sns.set_context(rc={"figure.figsize": (8, 4)})
    nd = np.arange(3)
    width = 0.8
    plt.xticks(nd + width / 2., xticks)
    plt.xlim(-0.15, 3)
    fig = plt.bar(nd, y, color=sns.color_palette("Blues", 3))
    plt.show()
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+'/norm_losses.png')
    plt.close()