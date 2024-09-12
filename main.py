import argparse
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import json
import os

from dataset.utils import prepare_cifar10_data
from model.vision_transformer import VisionTransformer
from vit_trainer import VitTrainer

import yaml


# parse YAML into a dictionary
def parse_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def get_args():
    parser = argparse.ArgumentParser(description="Vision Transformer model configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    return parser.parse_args()


def plot_run_metrics(experiment_name, base_dir="runs"):
    """
    Load and plot accuracy and loss for the given run.
    """
    metrics_file = os.path.join(base_dir, experiment_name, "metrics.json")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    train_losses = metrics['train_losses']
    test_losses = metrics['test_losses']
    accuracies = metrics['accuracies']

    epochs = range(1, len(train_losses) + 1)

    # plot training and validation Loss
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# plot the metrics for your run
plot_run_metrics("vit-default-cong")

if __name__ == "__main__":
    global optimiser, scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = get_args()
    config = parse_yaml_config(args.config)

    model_config = config["model_config"]
    training_config = config["training_config"]

    # configuration and model setup
    exp_name = 'vit-default-run-from-conf'
    batch_size = training_config['batch_size']
    epochs = training_config['epochs']
    lr = training_config['learning_rate']
    save_model_every = training_config['save_model_every']
    early_stopping_patience = training_config['early_stopping_patience']
    weight_decay = training_config['weight_decay']

    # init model, optimiser, scheduler, and loss function
    model = VisionTransformer(model_config)

    if training_config['optimiser'] == 'AdamW':
        optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif training_config['optimiser'] == 'SGD':
        optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if training_config['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10)
    elif training_config['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)

    loss_fn = nn.CrossEntropyLoss()

    # trainer instance
    trainer = VitTrainer(model, optimiser, loss_fn, exp_name, device, scheduler)

    # load data with the appropriate loaders
    trainloader, testloader, _ = prepare_cifar10_data(batch_size=batch_size)

    trainer.train(trainloader, testloader, epochs, save_model_every, early_stopping_patience)
