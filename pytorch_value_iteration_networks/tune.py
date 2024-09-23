import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from dataset.dataset import *
from utility.utils import *
from model import *
from ray import tune,train 
from ray.tune.schedulers import ASHAScheduler
from types import SimpleNamespace


class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, warmup_epochs=10, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.warmup_epochs = warmup_epochs  # Wait this many epochs before applying early stopping
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, epoch, val_loss, model):
        if epoch < self.warmup_epochs:
            return

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Save the model when validation loss improves.'''
        torch.save(model.state_dict(), "model_weights/" + self.path + ".pth")
        print(f"Model saved at epoch with validation loss: {self.best_loss:.4f}")


def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig('loss.png')
    plt.show()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_ray(config, trainloader, testloader, criterion):
    device = get_device()  # Select device based on availability
    net = VIN(SimpleNamespace(**config)).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=1e-5)

    #early_stopping = EarlyStopping(patience=10, delta=0.001, path='vin_full_traj')


    for epoch in range(config["epochs"]):
        avg_loss, num_batches = 0.0, 0.0
        for i, data in enumerate(trainloader):
            X, S1, S2, labels = [d.to(device) for d in data]
            if X.size()[0] != config["batch_size"]:
                continue

            optimizer.zero_grad()
            outputs, predictions = net(X, S1, S2, config["k"])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            num_batches += 1

        avg_loss /= num_batches
        val_loss = validate(net, testloader, config, criterion)

        # Report validation loss to Ray Tune
        train.report({"val_loss": val_loss})  # Report validation loss to Ray Tune

        # early_stopping(epoch, val_loss, net)
        # if early_stopping.early_stop:
        #     break


def validate(net: VIN, valloader, config, criterion):
    device = get_device()  # Select device based on availability
    val_loss = []

    for i, data in enumerate(valloader):
        X, S1, S2, labels = [d.to(device) for d in data]
        if X.size()[0] != config["batch_size"]:
            continue

        outputs, predictions = net(X, S1, S2, config["k"])
        loss = criterion(outputs, labels)
        val_loss.append(loss.item())

    return np.mean(val_loss)


def test(net: VIN, testloader, config):
    device = get_device()  # Select device based on availability
    total, correct = 0.0, 0.0

    for i, data in enumerate(testloader):
        X, S1, S2, labels = [d.to(device) for d in data]
        if X.size()[0] != config["batch_size"]:
            continue

        outputs, predictions = net(X, S1, S2, config["k"])
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        predicted = predicted.data
        correct += (torch.eq(torch.squeeze(predicted), labels)).sum()
        total += labels.size()[0]

    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))


def tune_train():
    print("Tuning hyperparameters...")
    config = {
        "lr": tune.loguniform(1e-4, 1e-2),  # Learning rate search space
        "batch_size": tune.choice([64]),  # Different batch sizes
        "epochs": 350,  # Large number of epochs; early stopping will handle termination
        "k": tune.choice([10, 16, 20, 30]),  # Number of Value Iterations
        "l_i": 2,  # Fixed input layer channels
        "l_h": 150,  # Fixed hidden layer channels
        "l_q": 4,  # Fixed Q-layer channels
    }

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=350,  # Max epochs, early stopping will likely stop before this
        grace_period=8,  # Minimum epochs before stopping
        reduction_factor=2  # Halve the number of trials each time
    )

    result = tune.run(
        tune.with_parameters(train_ray, trainloader=trainloader, testloader=testloader, criterion=nn.CrossEntropyLoss()),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=10,  # Number of hyperparameter configurations to try
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")


if __name__ == '__main__':
    # Parse training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default='/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/training_data/full_traj_vin_data.npz', help='Path to data file')
    parser.add_argument('--imsize', type=int, default=10, help='Size of image')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    config = parser.parse_args()

    transform = None

    trainset = GridworldData(config.datafile, imsize=config.imsize, train=True, transform=transform)
    testset = GridworldData(config.datafile, imsize=config.imsize, train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    tune_train()
