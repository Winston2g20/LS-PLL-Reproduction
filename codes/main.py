
from pathlib import Path

from prepare_data import *

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
        'AvgCL': [3, 4, 5]
    },
    {
        'Dataset': 'KuzushijiMNIST', 
        'Model': LeNet5, 
        'AvgCL': [3, 4, 5]
    },
    {
        'Dataset': 'CIFAR10', 
        'Model': ResNet18, 
        'AvgCL': [3, 4, 5]
    },
    {
        'Dataset': 'CIFAR100', 
        'Model': ResNet56, 
        'AvgCL': [7, 9, 11]
    }
]


def main():
    for exp in EXPERIMENTS:
        trainset, testset = load_dataset(exp['Dataset'])
        true_labels = np.array(trainset.targets)

        for avgCL in exp['AvgCL']:
            # Generate and load datasets
            model_path = f'../models/{exp['Dataset']}_{exp['Model'].name}.pth'
            if Path(model_path).exists():
                pass
            else:
                print(f"**** Training model for {exp['Dataset']} with {exp['Model'].name} ****")
                model = train_dataset_model(exp['Model'], trainset, testset)
                torch.save(model, model_path)
                print(f"**** Model saved to {model_path} ****")

            data_path = f'../datasets/pl_{exp['Dataset']}_avgcl{avgCL}.pkl'
            if Path(data_path).exists():
                pass
            else:
                print(f"**** Generating partial labels for {exp['Dataset']} with avgCL {avgCL} ****")
                predictions = get_topk_predictions(model, trainset)
                partial_labels = generate_partial_labels(true_labels, predictions, avg_num_cl=avgCL)

                with open(data_path, 'wb') as f:
                    pickle.dump(partial_labels, f)
                print(f"**** Partial labels saved to {data_path} ****")
        
            # Train model with partial labels
            pass

if __name__ == "__main__":
    main()