'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-06 16:42:21
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-09 18:13:32
FilePath: /LS-PLL-Reproduction/codes/main.py
Description: Main script containing the complete pipeline for training and evaluating models with partial labels.
'''

from pathlib import Path
import argparse
import os

from prepare_data import *
from LeNet5 import LeNet5
from ResNet18 import ResNet18
from ResNet56 import ResNet56
from train import LS_PLL_CrossEntropy, PartialLabelDataset, train_model
from utils import validate_path

# MODEL_PATH = './models'
# DATASET_PATH = './datasets'
parser = argparse.ArgumentParser(description='Full experiments')
parser.add_argument('--model_path', type=validate_path, default='./models', help="Path to the models folder")
parser.add_argument('--dataset_path', type=validate_path, default='./datasets', help="Path to the datasets folder")
args = parser.parse_args()
MODEL_PATH, DATASET_PATH = args.model_path, args.dataset_path

BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 200
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.9
SMOOTHING_RATE = [0.1, 0.3, 0.5, 0.7, 0.9]
EXPERIMENTS = [
    {
        'Dataset': 'FashionMNIST', 
        'Model': LeNet5, 
        'AvgCL': [3, 4, 5], 
        'NumClasses': 10, 
        'TopK': 6
    },
    {
        'Dataset': 'KuzushijiMNIST', 
        'Model': LeNet5, 
        'AvgCL': [3, 4, 5],
        'NumClasses': 10, 
        'TopK': 6
    },
    {
        'Dataset': 'CIFAR10', 
        'Model': ResNet18, 
        'AvgCL': [3, 4, 5],
        'NumClasses': 10, 
        'TopK': 6
    },
    {
        'Dataset': 'CIFAR100', 
        'Model': ResNet56, 
        'AvgCL': [7, 9, 11],
        'NumClasses': 100, 
        'TopK': 20
    }
]


def main():
    for exp in EXPERIMENTS: # for each models and relative datasets
        print()
        # load dataset
        trainset, testset = load_dataset(exp['Dataset'])
        if type(trainset.targets) == torch.Tensor:
            true_labels_train = trainset.targets.numpy()
            true_labels_test = testset.targets.numpy()
        else:
            true_labels_train = np.array(trainset.targets)
            true_labels_test = np.array(testset.targets)

        if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
        for avgCL in exp['AvgCL']: # for each noise levels
            # train model if model file not exist, or load model if exists
            model_path = f'{MODEL_PATH}/{exp['Dataset']}_{exp['Model'].name}.pth'
            if Path(model_path).exists():
                model = exp['Model'](num_classes=exp['NumClasses']).to(device)
                model.load_state_dict(torch.load(model_path))
                model.eval()
            else:
                print(f"**** Training model for {exp['Dataset']} with {exp['Model'].name} ****")
                model = train_dataset_model(exp['Model'], trainset, testset,
                                            num_classes=exp['NumClasses'])
                torch.save(model.state_dict(), model_path)
                print(f"**** Model saved to {model_path} ****")

            # load, or generate and load partial datasets for both train and test sets
            if not os.path.exists(DATASET_PATH): os.makedirs(DATASET_PATH)
            traindata_path = f'{DATASET_PATH}/pl_{exp['Dataset']}_avgcl{avgCL}_train.npy'
            if Path(traindata_path).exists(): partial_labels_train = np.load(traindata_path)
            else:
                print(f"**** Generating partial labels for {exp['Dataset']} with avgCL {avgCL} ****")
                predictions_train = get_topk_predictions(model, trainset, k=exp['TopK'])
                partial_labels_train, _ = generate_partial_labels(true_labels_train, predictions_train, 
                                                                avg_cl=avgCL, k=exp['TopK'], 
                                                                num_classes=exp['NumClasses'])
                np.save(traindata_path, partial_labels_train)
                print(f"**** Partial labels saved to {traindata_path} ****")

            testdata_path = f'{DATASET_PATH}/pl_{exp['Dataset']}_avgcl{avgCL}_test.npy'
            if Path(testdata_path).exists(): partial_labels_test = np.load(testdata_path)
            else:
                predictions_test = get_topk_predictions(model, testset, k=exp['TopK'])
                partial_labels_test, _ = generate_partial_labels(true_labels_test, predictions_test, 
                                                                avg_cl=avgCL, k=exp['TopK'], 
                                                                num_classes=exp['NumClasses'])
                np.save(testdata_path, partial_labels_test)
                print(f"**** Partial labels saved to {testdata_path} ****")

            # Train model with partial labels without label smoothing
            # using nn.CrossEntropy as default.
            trainset = PartialLabelDataset(trainset, partial_labels_train, transform=transforms.ToTensor())
            testset = PartialLabelDataset(testset, partial_labels_test, transform=transforms.ToTensor())
            print(f"**** Training on partial labelled {exp['Dataset']} without label smoothing ****")
            _, non_smoothing_record = train_model(exp['Model'], trainset, testset, 
                                                num_epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                                lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, 
                                                num_classes=exp['NumClasses'], label_format='multihot')

            # train model with label smoothing across different smoothing rates
            smoothing_records = {}
            for r in SMOOTHING_RATE:
                print(f"\n**** Training on partial labelled {exp['Dataset']} with a smoothing rate of {r} ****")
                _, record = train_model(exp['Model'], trainset, testset, 
                            num_epochs=EPOCHS, batch_size=BATCH_SIZE, 
                            lr=LEARNING_RATE, momentum=MOMENTUM, 
                            weight_decay=WEIGHT_DECAY, num_classes=exp['NumClasses'],
                            criterion=LS_PLL_CrossEntropy(smoothing_rate=r).to(device), 
                            label_format='multihot')
                smoothing_records[r] = record

if __name__ == "__main__":
    main()
