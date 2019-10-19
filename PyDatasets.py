import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

DATASET_FOLDER = 'data

def pad_image(padding=1, fill=0, padding_mode='constant'):
    return transforms.Pad(padding, fill, padding_mode)

def normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    """normalizes image[CxHxW] where
        output[channel] = (input[channel] - m"""
    return transforms.Normalize(mean, std)

def toTensor():
    """converts PIL image[HxWxC] in the range [0,255]
        to floatTensor [CxHxW] in range [0.0,0.1]"""
    return transforms.ToTensor()


def get_cifar10(train=False, transform=None, download=True):

    return datasets.CIFAR10(root=os.path.join(DATASET_FOLDER, "CIFAR10"),
                            train=train,
                            transform=transform,
                            download=download)

def get_cifar100(train=False, transform=None, download=True):

    return datasets.CIFAR100(root=os.path.join(DATASET_FOLDER, "CIFAR100"),
                            train=train,
                            transform=transform,
                            download=download)

def get_svhn(train=False, transform=None, download=True):

    return datasets.SVHN(root=os.path.join(DATASET_FOLDER, "SVHN"),
                            train=train,
                            transform=transform,
                            download=download)

