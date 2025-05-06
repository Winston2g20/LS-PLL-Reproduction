import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
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
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class ResNet56(nn.Module):
    def __init__(self, num_classes):
        super(ResNet56, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def load_dataset(dataset_name='CIFAR10'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
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
    k=6, num_classes=10
):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    train_model = model(num_classes=num_classes).to(device)
    optimizer = optim.SGD(train_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
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

    train_model.eval()
    with torch.no_grad():
        topk_preds = []
        for inputs, labels in DataLoader(trainset, batch_size=batch_size):
            outputs = train_model(inputs.to(device))
            _, topk = outputs.topk(k+1, dim=1)
            topk_preds.append(topk.cpu().numpy())
        topk_preds = np.concatenate(topk_preds, axis=0)

    return topk_preds, train_model


if __name__ == "__main__":
    trainset, testset = load_dataset()
    true_labels = np.array(trainset.targets)

    predictions, _ = train_dataset_model(ResNet18, trainset, testset)
    partial_labels = generate_partial_labels(true_labels, predictions, avg_num_cl=5)

    with open('../datasets/pl_cifar10_avgcl5.pkl', 'wb') as f:
        pickle.dump(partial_labels, f)