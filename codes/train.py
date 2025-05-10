'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-09 15:22:32
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-10 18:39:34
FilePath: /LS-PLL-Reproduction/codes/train.py
Description: Functions relates to model training
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from PIL import Image

from utils import device


class LS_PLL_CrossEntropy(nn.Module):
    def __init__(self, smoothing_rate=0.1, ema_decay=0.9):
        super(LS_PLL_CrossEntropy, self).__init__()
        self.smoothing_rate = smoothing_rate
        self.eta = ema_decay
        self.register_buffer('ema_probs', torch.tensor([]))

    def forward(self, logits, candidates):
        """
        Partial Label Learning with Smoothing Cross Entropy
        params:
            logits (Tensor): model's output logits [batch_size, num_classes]
            candidates (Tensor): multi-hot encoded labels [batch_size, num_classes]
        return:
            loss (Tensor): cross-entropy loss after smoothing
        """
        # initialise EMA buffer
        batch_size, num_classes = logits.size()
        if self.ema_probs.numel() != batch_size * num_classes:
            self.ema_probs = torch.zeros_like(logits)

        # calc softmax and discard non candidates
        probs = F.softmax(logits, dim=1) * candidates

        # update EMA (η)
        # q^(t) = η q^(t-1) + (1-η) q
        self.ema_probs = self.eta * self.ema_probs + (1 - self.eta) * probs

        # normalise candidate set and estimate pseudo-true label
        denom = (self.ema_probs * candidates).sum(dim=1, keepdim=True).clamp(min=1e-12)
        normalized = (self.ema_probs * candidates) / denom
        true_label_idx = normalized.argmax(dim=1)

        # one_hot[j] = 1 iff j = y_i^(t)
        one_hot = torch.zeros_like(candidates).scatter_(1, true_label_idx.unsqueeze(1), 1.0)

        # calculate size of candidates of each sample
        set_size = candidates.sum(dim=1, keepdim=True).clamp(min=1).float()

        # label smoothing
        # within candidate set: Y_ls_j = (1-r) * I[j=y] + (r / |Y|)
        # outside candidate set: Y_ls_j = 0
        Y_ls = (1 - self.smoothing_rate) * one_hot + (self.smoothing_rate / set_size) * candidates

        # calc cross entropy loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(Y_ls * log_probs).sum(dim=1).mean()
        return loss


class PartialLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, partial_labels, 
                transform=None, target_transform=None):
        self.data = dataset.data
        self.targets = partial_labels.astype(float)

        self.transform = transform
        self.target_transform = target_transform

        transforms = datasets.vision.StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if type(img) == torch.Tensor:
            img = img.numpy()

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def train_model(
    Model, trainset, testset, 
    num_epochs=200, batch_size=128,
    lr=0.01, momentum=0.9, 
    num_classes=10, criterion=nn.CrossEntropyLoss(),
    label_format='indices'
):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model = Model(num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    test_criterion = nn.CrossEntropyLoss()
    records = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = correct = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            if label_format == 'multihot':
                batch_indices = torch.arange(predictions.size(0), device=device)
                correct += labels[batch_indices, predictions].sum().item()
            else: correct += (predictions == labels).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total * 100

        model.eval()
        with torch.no_grad():
            running_test_loss = 0.0
            total = correct = 0
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = test_criterion(outputs, labels)

                running_test_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                if label_format == 'multihot':
                    batch_indices = torch.arange(predictions.size(0), device=device)
                    correct += labels[batch_indices, predictions].sum().item()
                else: correct += (predictions == labels).sum().item()
                total += labels.size(0)

            test_loss = running_test_loss / len(testloader)
            test_acc = correct / total * 100

        records['train_loss'].append(train_loss)
        records['train_acc'].append(train_acc)
        records['val_loss'].append(test_loss)
        records['val_acc'].append(test_acc)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: \
                    \n\tTrain Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.3f} \
                    \n\tTest Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.3f}')

    return model, records

