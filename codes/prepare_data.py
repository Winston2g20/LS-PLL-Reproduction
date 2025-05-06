'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-06 15:24:13
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-06 21:52:44
FilePath: /LS-PLL-Reproduction/codes/prepare_data.py
Description: The codes to download, train and generate partial labels for datasets.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.resnet import BasicBlock

import numpy as np
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


class LeNet5(nn.Module):
    name = 'LeNet5'
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    name = 'ResNet18'
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        x = self.relu(x)
        return x
class ResNet56(nn.Module):
    name = 'ResNet56'
    def __init__(self, num_classes):
        super(ResNet56, self).__init__()
        
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(Bottleneck, 16, 9, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 32, 9, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 64, 9, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_dataset(dataset_name='CIFAR10'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    print(f"**** Loading {dataset_name} dataset ****")
    if dataset_name == 'CIFAR10':
        trainset = datasets.CIFAR10(root=f'../datasets/{dataset_name}', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=f'../datasets/{dataset_name}', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        trainset = datasets.CIFAR100(root=f'../datasets/{dataset_name}', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root=f'../datasets/{dataset_name}', train=False, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        trainset = datasets.FashionMNIST(root=f'../datasets/{dataset_name}', train=True, download=True, transform=transforms.ToTensor())
        testset = datasets.FashionMNIST(root=f'../datasets/{dataset_name}', train=False, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'KuzushijiMNIST':
        trainset = datasets.KMNIST(root=f'../datasets/{dataset_name}', train=True, download=True, transform=transforms.ToTensor())
        testset = datasets.KMNIST(root=f'../datasets/{dataset_name}', train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return trainset, testset


def generate_partial_labels(true_labels, topk_preds, avg_num_cl):
    num_samples = len(true_labels)
    candidate_sizes = np.random.poisson(lam=avg_num_cl-1, size=num_samples) + 1
    candidate_sizes = np.clip(candidate_sizes, 1, topk_preds.shape[1])
    
    partial_labels = []
    for i in range(num_samples):
        true_label = true_labels[i]
        valid_preds = [p for p in topk_preds[i] if p != true_label]
        noise_labels = valid_preds[:candidate_sizes[i]-1]
        candidate_set = [true_label] + noise_labels
        partial_labels.append(candidate_set)
    return partial_labels


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
        for inputs, _ in DataLoader(dataset, batch_size=batch_size):
            outputs = model(inputs.to(device))
            _, topk = outputs.topk(k+1, dim=1)
            topk_preds.append(topk.cpu().numpy())
        topk_preds = np.concatenate(topk_preds, axis=0)
    return topk_preds

if __name__ == "__main__":
    trainset, testset = load_dataset()
    true_labels = np.array(trainset.targets)

    model = train_dataset_model(ResNet18, trainset, testset)
    predictions = get_topk_predictions(model, trainset)
    partial_labels = generate_partial_labels(true_labels, predictions, avg_num_cl=5)

    with open('../datasets/pl_CIFAR10_avgcl5.pkl', 'wb') as f:
        pickle.dump(partial_labels, f)