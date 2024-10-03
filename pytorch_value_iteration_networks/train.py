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
import os


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
            # Don't apply early stopping during the warm-up period
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
        torch.save(model.state_dict(),"../model_weights/" + self.path+".pth")
        print(f"Model saved at epoch with validation loss: {self.best_loss:.4f}")

    
def plot_loss(train_losses,val_losses):
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig('loss.png')



def train(net: VIN, trainloader, testloader,config, criterion, optimizer):
    print_header()
    # Automatically select device to make the code device agnostic
    # device = torch.device("cuda:0" if torch.cuda.is_available() elif torch)
 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    early_stopping = EarlyStopping(patience=10, delta=0.001, path='vin_all_obs')
    train_losses = []
    val_losses = []
    for epoch in range(config.epochs):  # Loop over dataset multiple times
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, S1, S2, labels = [d.to(device) for d in data]
            if X.size()[0] != config.batch_size:
                continue  # Drop those data, if not enough for a batch
            net = net.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs, predictions = net(X, S1, S2, config.k)
            # Loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch, error_batch = get_stats(loss, predictions, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            num_batches += 1
        time_duration = time.time() - start_time 
        # Print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)
        # Validate
        train_losses.append(avg_loss/num_batches)
        
        val_loss = validate(net, testloader, config, criterion)

        val_losses.append(val_loss)

        early_stopping(epoch, val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    
    plot_loss(train_losses,val_losses)
    print('\nFinished training. \n')


def validate(net: VIN, valloader, config, criterion):
    total, correct = 0.0, 0.0
    # Automatically select device, device agnostic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    val_loss = []  # Validation loss

    for i, data in enumerate(valloader):
        X, S1, S2, labels = [d.to(device) for d in data]
        if X.size()[0] != config.batch_size:
            continue  # Drop those data, if not enough for a batch
        net = net.to(device)
        # Forward pass
        outputs, predictions = net(X, S1, S2, config.k)

        loss = criterion(outputs, labels)

        val_loss.append(loss.item())
    
    return np.mean(val_loss)

        






def test(net: VIN, testloader, config):
    total, correct = 0.0, 0.0
    # Automatically select device, device agnostic
     
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, labels = [d.to(device) for d in data]
        if X.size()[0] != config.batch_size:
            continue  # Drop those data, if not enough for a batch
        net = net.to(device)
        # Forward pass
        outputs, predictions = net(X, S1, S2, config.k)
        # Select actions with max scores(logits)
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data
        # Compute test accuracy
        correct += (torch.eq(torch.squeeze(predicted), labels)).sum()
        total += labels.size()[0]
    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))



def parse_args():
       # Handle file-based arguments
    new_args = []
    for arg in sys.argv[1:]:
        if arg.startswith('@'):
            with open(arg[1:], 'r') as f:
                # Add each line in the file to the arguments list, splitting by whitespace
                new_args.extend(f.read().split())
        else:
            new_args.append(arg)


if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datafile',
        type=str,
        default='/media/vanderbilt/home/nkepling/fullyObservableNavigation/training_data/diverse_traj.npz',
        help='Path to data file')
    parser.add_argument('--imsize', type=int, default=10, help='Size of image')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument(
        '--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument(
        '--k', type=int, default=16, help='Number of Value Iterations')
    parser.add_argument(
        '--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument(
        '--l_h',
        type=int,
        default=150,
        help='Number of channels in first hidden layer')
    parser.add_argument(
        '--l_q',
        type=int,
        default=4,
        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    config = parser.parse_args()
    # Get path to save trained model
    # save_path = "trained/vin_{0}x{0}.pth".format(config.imsize)
    save_path = "trained/vin_all_obs_1.pth"
    # Instantiate a VIN model
    net = VIN(config)
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    # optimizer = optim.RMSprop(net.parameters(), lr=config.lr, eps=1e-6)
    optimizer = optim.Adam(net.parameters(),lr=config.lr,weight_decay=1e-5)
    # Dataset transformer: torchvision.transforms
    transform = None
    # Define Dataset
    trainset = GridworldData(
        config.datafile, imsize=config.imsize, train=True, transform=transform)
    testset = GridworldData(
        config.datafile,
        imsize=config.imsize,
        train=False,
        transform=transform)
    # Create Dataloader

    # print(len(trainset))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=0)


    
    #Train the model
    train(net, trainloader, testloader, config, criterion, optimizer)
    # Test accuracy
    test(net, testloader, config)
    # Save the trained model parameters
    torch.save(net.state_dict(), save_path)
