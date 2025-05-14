import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import pickle

from ResNet18 import ResNet18
from utils import device, seed
from utils import validate_path, extract_features, tsne_plot, plot_grid


def load_dataset(dataset_name='CIFAR10', dataset_path='../datasets'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    print(f"**** Loading {dataset_name} dataset ****")
    if dataset_name == 'CIFAR10':
        trainset = datasets.CIFAR10(root=f"{dataset_path}/{dataset_name}", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=f"{dataset_path}/{dataset_name}", train=False, download=True, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        trainset = datasets.CIFAR100(root=f"{dataset_path}/{dataset_name}", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=f"{dataset_path}/{dataset_name}", train=False, download=True, transform=transform_test)
    elif dataset_name == 'FashionMNIST':
        trainset = datasets.FashionMNIST(root=f"{dataset_path}/{dataset_name}", train=True, download=True, transform=transform_train)
        testset = datasets.FashionMNIST(root=f"{dataset_path}/{dataset_name}", train=False, download=True, transform=transform_test)
    elif dataset_name == 'KuzushijiMNIST':
        trainset = datasets.KMNIST(root=f"{dataset_path}/{dataset_name}", train=True, download=True, transform=transform_train)
        testset = datasets.KMNIST(root=f"{dataset_path}/{dataset_name}", train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return trainset, testset


def main():
    load_dataset('CIFAR10')
    load_dataset('CIFAR100')
    load_dataset('KuzushijiMNIST')

if __name__ == "__main__":
    main()
