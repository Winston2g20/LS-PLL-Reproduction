'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-06 16:42:21
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-12
FilePath: /LS-PLL-Reproduction/codes/main.py
Description: Main script containing the complete pipeline for training and evaluating models with partial labels.
'''

from pathlib import Path
import argparse
import os
import torch
import numpy as np
import pickle
import torchvision.transforms as transforms

from prepare_data import *
from LeNet5 import LeNet5
from ResNet18 import ResNet18
from ResNet56 import ResNet56
from train import LS_PLL_CrossEntropy, PartialLabelDataset, train_model
from utils import validate_path, extract_features, tsne_plot, plot_grid

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser
parser = argparse.ArgumentParser(description='Full experiments')
parser.add_argument('--model_path', type=validate_path, default='./models', help="Path to the models folder")
parser.add_argument('--dataset_path', type=validate_path, default='./datasets', help="Path to the datasets folder")
parser.add_argument('--figure_path', type=validate_path, default='./doc/figures', help="Path to the figures folder")
args = parser.parse_args()
MODEL_PATH, DATASET_PATH, FIGURE_PATH = args.model_path, args.dataset_path, args.figure_path

# Hyperparameters and experiment configurations
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 200
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.9
WEIGHTING_PARAM = 0.9
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
        models, records = {}, {}
        figure_paths, titles = [], []
        plot_idx = 0

        print(f"\n===== Starting experiment for {exp['Dataset']} with {exp['Model'].__name__} =====")
        trainset, testset = load_dataset(exp['Dataset'], dataset_path=DATASET_PATH)
        true_labels_train = np.array(trainset.targets)
        true_labels_test = np.array(testset.targets)

        for avgCL in exp['AvgCL']:
            models[avgCL], records[avgCL] = [], []

            # Train/load dataset model
            dataset_model_path = f"{MODEL_PATH}/PL_{exp['Dataset']}_{exp['Model'].__name__}.pth"
            if Path(dataset_model_path).exists():
                model = exp['Model'](num_classes=exp['NumClasses']).to(device)
                model.load_state_dict(torch.load(dataset_model_path))
                model.eval()
            else:
                print(f"**** Training model for {exp['Dataset']} with {exp['Model'].__name__} ****")
                model = train_dataset_model(exp['Model'], trainset, testset, num_classes=exp['NumClasses'])
                torch.save(model.state_dict(), dataset_model_path)
                print(f"**** Model saved to {dataset_model_path} ****")

            # Generate/load partial labels for train set
            traindata_path = f"{DATASET_PATH}/pl_{exp['Dataset']}_avgcl{avgCL}_train.npy"
            if Path(traindata_path).exists():
                partial_labels_train = np.load(traindata_path)
            else:
                predictions_train = get_random_predictions(model, trainset, k=exp['TopK'])
                partial_labels_train, _ = generate_partial_labels(true_labels_train, predictions_train,
                                                                  avg_cl=avgCL, k=exp['TopK'],
                                                                  num_classes=exp['NumClasses'])
                np.save(traindata_path, partial_labels_train)
                print(f"**** Partial labels saved to {traindata_path} ****")

            # Generate/load partial labels for test set
            testdata_path = f"{DATASET_PATH}/pl_{exp['Dataset']}_avgcl{avgCL}_test.npy"
            if Path(testdata_path).exists():
                partial_labels_test = np.load(testdata_path)
            else:
                predictions_test = get_random_predictions(model, testset, k=exp['TopK'])
                partial_labels_test, _ = generate_partial_labels(true_labels_test, predictions_test,
                                                                 avg_cl=avgCL, k=exp['TopK'],
                                                                 num_classes=exp['NumClasses'])
                np.save(testdata_path, partial_labels_test)
                print(f"**** Partial labels saved to {testdata_path} ****")

            # Train without label smoothing
            model_dir = f"{MODEL_PATH}/{exp['Model'].__name__}/avgcl{avgCL}"
            os.makedirs(model_dir, exist_ok=True)

            train_set = PartialLabelDataset(trainset, partial_labels_train, transform=transforms.ToTensor())
            test_set = PartialLabelDataset(testset, partial_labels_test, transform=transforms.ToTensor())

            print(f"**** Training {exp['Model'].__name__} on {exp['Dataset']} (Avg.#CL={avgCL}) without LS ****")
            non_smoothing_model, non_smoothing_record = train_model(
                exp['Model'], train_set, test_set, num_epochs=EPOCHS,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE, momentum=MOMENTUM,
                weighting_param=WEIGHTING_PARAM, num_classes=exp['NumClasses']
            )
            models[avgCL].append(non_smoothing_model)
            records[avgCL].append(non_smoothing_record)
            torch.save(non_smoothing_model.state_dict(), f"{model_dir}/r_noLS.pth")
            print(f"**** Model saved to {model_dir}/r_noLS.pth ****")

            # t-SNE plot
            figure_path = f"{FIGURE_PATH}/tsne_{exp['Dataset']}_cl{avgCL}_r_noLSRandom.png"
            figure_paths.append(figure_path)
            titles.append(f"({chr(97+plot_idx)}) w/o LS")
            plot_idx += 1
            features, labels = extract_features(non_smoothing_model, testset, batch_size=BATCH_SIZE)
            tsne_plot(features, labels, figure_path, f"Avg.#CL={avgCL}")
            print(f"**** TSNE plot saved to {figure_path} ****")

            # Train with label smoothing
            for r in SMOOTHING_RATE:
                print(f"\n**** Training {exp['Model'].__name__} on {exp['Dataset']} (Avg.#CL={avgCL}) with LS rate {r} ****")
                model, record = train_model(
                    exp['Model'], train_set, test_set, num_epochs=EPOCHS,
                    batch_size=BATCH_SIZE, lr=LEARNING_RATE, momentum=MOMENTUM,
                    weighting_param=WEIGHTING_PARAM, num_classes=exp['NumClasses'],
                    smoothing_rate=r
                )
                models[avgCL].append(model)
                records[avgCL].append(record)
                torch.save(model.state_dict(), f"{model_dir}/r_{r}.pth")
                print(f"**** Model saved to {model_dir}/r_{r}.pth ****")

                figure_path = f"{FIGURE_PATH}/tsne_{exp['Dataset']}_cl{avgCL}_r_{r}Random.png"
                figure_paths.append(figure_path)
                titles.append(f"({chr(97+len(titles))}) w/ LS, r={r}")
                plot_idx += 1
                features, labels = extract_features(model, testset, batch_size=BATCH_SIZE)
                tsne_plot(features, labels, figure_path)
                print(f"**** TSNE plot saved to {figure_path} ****")

            # Save training records
            with open(f"{model_dir}/records.pkl", 'wb') as f:
                pickle.dump(records, f)

        # Grid plot
        plot_grid(
            figure_paths, titles,
            rows=len(exp['AvgCL']),
            cols=len(SMOOTHING_RATE) + 1,
            save_path=f"{FIGURE_PATH}/tsne_grid_{exp['Dataset']}.png"
        )


if __name__ == "__main__":
    main()
