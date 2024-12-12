from nn_training import *  # Ensure AutoEncoderDataset, Unet, and train_model are in nn_training
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import os
import pickle
from dl_models import UNet
from torch.utils.data import DataLoader


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural Network Training parameters")

    parser.add_argument('--exp_name', type=str, help='experiment name', required=True)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=100)
    parser.add_argument('--DataPath', type=str, help='path to training data', required=True)
    parser.add_argument('--ObstacleMap', type=str, help='path to saved obstacle map', default="obstacle.pkl")
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=64)
    parser.add_argument('--train_split', type=float, help='train set size as a percentage', default=0.70)  # Changed the default to 70%
    parser.add_argument('--val_split', type=float, help='validation set size as a percentage', default=0.15)
    parser.add_argument('--test_split', type=float, help='test set size as a percentage', default=0.15)
    parser.add_argument('--seed', type=int, help='set seed for training', default=42)
    parser.add_argument('--delta',type=float, help='improvement delta for early stopping', default=0.0001)

    args, unknown_args = parser.parse_known_args()  # Handle unknown args gracefully

    # Output the arguments to verify if they're being captured properly
    print(f'Experiment Name: {args.exp_name}')
    print(f'Learning Rate: {args.lr}')
    print(f'Epochs: {args.epochs}')
    print(f'DataPath: {args.DataPath}')
    print(f'ObstacleMap: {args.ObstacleMap}')
    print(f'Batch Size: {args.batch_size}')
    print(f'Train Split: {args.train_split}')
    print(f'Val Split: {args.val_split}')
    print(f'Test Split: {args.test_split}')
    print(f'Seed: {args.seed}')
    print(f'Delta: {args.delta}')

    # Save the arguments to a file
    with open(f'{args.exp_name}_args_saved.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')

    # Load the obstacle map
    with open(args.ObstacleMap, "rb") as f:
        obstacle_map = pickle.load(f)

    # Set random seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Prepare dataset and model
    data_dir = args.DataPath
    model_name = args.exp_name

    dataset_size = len([os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(('pt'))])

    train_size = int(args.train_split * dataset_size)
    val_size = int(args.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_ind, test_ind, val_ind = torch.utils.data.random_split(range(dataset_size), [train_size, test_size, val_size])

    train_data = AutoEncoderDataset(data_dir, train_ind)
    val_data = AutoEncoderDataset(data_dir, val_ind)
    test_data = AutoEncoderDataset(data_dir, test_ind)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    model_path = args.exp_name
    num_epochs = args.epochs  # Access args.epochs instead of args.num_epochs

    criterion = nn.MSELoss()

    model = UNet()  # Ensure this is imported correctly

    # Train the model using the training function
    train_model(train_dataloader, val_dataloader, model, model_path, num_epochs, criterion,args.delta)
