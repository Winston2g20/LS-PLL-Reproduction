'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-06 16:42:21
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-08 22:58:01
FilePath: /LS-PLL-Reproduction/codes/main.py
Description: Main script containing the complete pipeline for training and evaluating models with partial labels.
'''

from pathlib import Path
import os

from prepare_data import *
from LeNet5 import LeNet5
from ResNet18 import ResNet18
from ResNet56 import ResNet56

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
    for exp in EXPERIMENTS:
        print()
        trainset, testset = load_dataset(exp['Dataset'])
        true_labels_train = trainset.targets.numpy()
        true_labels_test = testset.targets.numpy()

        if not os.path.exists('../models'): os.makedirs('../models')
        for avgCL in exp['AvgCL']:
            # Generate and load datasets
            model_path = f'../models/{exp['Dataset']}_{exp['Model'].name}.pth'
            if not Path(model_path).exists():
                print(f"**** Training model for {exp['Dataset']} with {exp['Model'].name} ****")
                model = train_dataset_model(exp['Model'], trainset, testset,
                                            num_classes=exp['NumClasses'])
                torch.save(model.state_dict(), model_path)
                print(f"**** Model saved to {model_path} ****")
            else:
                model = exp['Model'](num_classes=exp['NumClasses']).to(device)
                model.load_state_dict(torch.load(model_path))
                model.eval()

            if not os.path.exists('../datasets'): os.makedirs('../datasets')
            traindata_path = f'../datasets/pl_{exp['Dataset']}_avgcl{avgCL}_train.npy'
            if not Path(traindata_path).exists():
                print(f"**** Generating partial labels for {exp['Dataset']} with avgCL {avgCL} ****")
                predictions_train = get_topk_predictions(model, trainset, k=exp['TopK'])
                partial_labels_train = generate_partial_labels(true_labels_train, predictions_train, avg_cl=avgCL, k=exp['TopK'])
                np.save(traindata_path, partial_labels_train)
                print(f"**** Partial labels saved to {traindata_path} ****")
            else: partial_labels_train = np.load(traindata_path)

            testdata_path = f'../datasets/pl_{exp['Dataset']}_avgcl{avgCL}_test.npy'
            if not Path(testdata_path).exists():
                predictions_test = get_topk_predictions(model, testset, k=exp['TopK'])
                partial_labels_test = generate_partial_labels(true_labels_test, predictions_test, avg_cl=avgCL, k=exp['TopK'])
                np.save(testdata_path, partial_labels_test)
                print(f"**** Partial labels saved to {testdata_path} ****")
            else: partial_labels_train = np.load(testdata_path)

            # Train model with partial labels

if __name__ == "__main__":
    main()
