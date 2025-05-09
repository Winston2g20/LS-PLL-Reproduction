'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-06 15:24:13
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-09 00:31:06
FilePath: /LS-PLL-Reproduction/codes/prepare_data.py
Description: The codes to download, train and generate partial labels for datasets.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import pickle

from ResNet18 import ResNet18

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def load_dataset(dataset_name='CIFAR10'):
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
        trainset = datasets.CIFAR10(root=f'../datasets/{dataset_name}', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=f'../datasets/{dataset_name}', train=False, download=True, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        trainset = datasets.CIFAR100(root=f'../datasets/{dataset_name}', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=f'../datasets/{dataset_name}', train=False, download=True, transform=transform_test)
    elif dataset_name == 'FashionMNIST':
        trainset = datasets.FashionMNIST(root=f'../datasets/{dataset_name}', train=True, download=True, transform=transform_train)
        testset = datasets.FashionMNIST(root=f'../datasets/{dataset_name}', train=False, download=True, transform=transform_test)
    elif dataset_name == 'KuzushijiMNIST':
        trainset = datasets.KMNIST(root=f'../datasets/{dataset_name}', train=True, download=True, transform=transform_train)
        testset = datasets.KMNIST(root=f'../datasets/{dataset_name}', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return trainset, testset


def train_dataset_model(
    model, trainset, testset, 
    num_epochs=200, batch_size=128,
    lr=0.01, momentum=0.9, weight_decay=1e-3,
    num_classes=10
):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    train_model = model(num_classes=num_classes).to(device)
    optimizer = optim.SGD(train_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_model.train()
        running_loss = 0.0
        total = correct = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = train_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total * 100

        train_model.eval()
        with torch.no_grad():
            test_loss = 0.0
            total = correct = 0
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = train_model(inputs)
                loss = loss_fn(outputs, labels)

                test_loss = loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            test_loss /= len(testloader)
            test_acc = correct / total * 100
    
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: \
                    \n\tTrain Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.3f} \
                    \n\tTest Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.3f}')

    return train_model


def get_topk_predictions(model, dataset, batch_size=128, k=6):
    model.eval()
    with torch.no_grad():
        topk_preds = []
        for inputs, targets in DataLoader(dataset, batch_size=batch_size):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # remove the true label from the top-k predictions
            # by setting the true label's score to -inf
            rows = np.arange(outputs.shape[0])
            masked_outputs = outputs.clone()
            masked_outputs[rows, targets] = -np.inf
            # get the top-k predictions
            _, topk = masked_outputs.topk(k, dim=1)

            topk_preds.append(topk.cpu().numpy())
        topk_preds = np.concatenate(topk_preds, axis=0)
    return topk_preds


def generate_partial_labels(true_labels, topk_preds, avg_cl, k=6, num_classes=10):
    n = len(true_labels)
    cl_values = np.random.randint(0, k, size=n)
    current_sum = cl_values.sum()
    target_sum = avg_cl * n

    while abs(current_sum - target_sum) > 0.01 * n:
        idx = np.random.randint(0, n)
        diff = target_sum - current_sum
        if diff > 0 and cl_values[idx] < k:
            cl_values[idx] += 1
            current_sum += 1
        elif diff < 0 and cl_values[idx] > 0:
            cl_values[idx] -= 1
            current_sum -= 1
    
    random_mask = np.argsort(np.random.rand(n, k), axis=1) < cl_values.reshape(-1, 1)
    partial_labels = [topk_preds[i, mask] for i, mask in enumerate(random_mask)]
    partial_labels = [np.append(arr, gt) for arr, gt in zip(true_labels, partial_labels)]

    onehot_labels = np.zeros((n, num_classes), dtype=np.bool)
    for i in range(n):
        onehot_labels[i, partial_labels[i]] = 1
    return onehot_labels, partial_labels


if __name__ == "__main__":
    trainset, testset = load_dataset()
    true_labels = np.array(trainset.targets)

    model = train_dataset_model(ResNet18, trainset, testset)
    predictions = get_topk_predictions(model, trainset)
    partial_labels = generate_partial_labels(true_labels, predictions, avg_num_cl=5)

    with open('../datasets/pl_CIFAR10_avgcl5.pkl', 'wb') as f:
        pickle.dump(partial_labels, f)
