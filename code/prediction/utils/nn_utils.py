import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import Any


class SiameseNN(nn.Module):
    def __init__(self, params):
        super(SiameseNN, self).__init__()
        self.base_network = create_base_network(params)

    def forward(self, x1, x2):
        h1 = self.base_network(x1)
        h2 = self.base_network(x2)
        delta_h = torch.abs(h1 - h2)
        dh_size = delta_h.size(1)
        d_logit = nn.Linear(dh_size, 1)(delta_h)
        d_sigmoid = nn.Sigmoid()(d_logit)
        return d_sigmoid


def get_activation(params):
    activation_map = {
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(),
        'ELU': nn.ELU()
    }
    return activation_map[params['activation']]

def get_scheduler(optimizer, params):
    scheduler_map = {
        'linear':  torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=params['num_epochs']),
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs']),
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params['patience'])
    }
    """
    'exp': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma']),
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params['patience']),
    'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['num_epochs']//params['num_steps'],
                                            gamma=1 - params['gamma'])
    """
    return scheduler_map[params['scheduler']]



def should_stop(loss: list, patience: int=3) -> bool:
    """
    implementation of early stopping
    :param loss: loss function list
    :param patience: if the loss rises for `patience` losses, we stop
    :return: increasing: if true, we should stop
    """
    if len(loss) <= patience:
        return False
    recent_loss = loss[np.maximum(0, len(loss) - patience):]
    increasing = all(x < y for x, y in zip(recent_loss, recent_loss[1:]))
    return increasing

def train_model_nn(X_train_1: torch.Tensor, X_train_2: torch.Tensor, y_train: torch.Tensor,
                   X_val_1: torch.Tensor, X_val_2: torch.Tensor, y_val: torch.Tensor,
                   params: dict, check_val: bool=False) -> tuple[float, Any, Any]:



    # Standardizing data

    # scaler = StandardScaler()
    # scaler.fit(torch.cat((X_train_1, X_train_2), dim=0))
    # X_train_1_scaled = scaler.transform(X_train_1)
    # X_train_2_scaled = scaler.transform(X_train_2)
    # X_val_1_scaled = scaler.transform(X_val_1)
    # X_val_2_scaled = scaler.transform(X_val_2)
    # Transforming data to torch tensors

    # Creating datasets
    train_dataset = TensorDataset(X_train_1, X_train_2, y_train)

    # Defining model
    model = SiameseNN(params)

    # Defining training parameters
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = get_scheduler(optimizer, params)

    epoch_losses = []
    val_losses = []

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    # Training loop
    for epoch in range(params['num_epochs']):
        model.train()
        epoch_loss = 0
        curr_batch = 0
        # for each batch in the epoch
        for X_1_batch, X_2_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_1_batch, X_2_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_1_batch.size(0)/len(train_loader.dataset)
            curr_batch += 1
            print(f'batch: {curr_batch} out of {len(train_loader)}. loss: {loss.item()}')
        epoch_losses.append(epoch_loss)

        # Checking validation loss for this epoch
        if check_val:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_1, X_val_2)
                val_loss = criterion(y_val_pred, y_val)
                val_losses.append(val_loss)

            # Early stopping
            if should_stop(val_losses):
                break

        print(f'epoch: {epoch}, loss: {np.mean(epoch_losses)}, val_loss: {np.mean(val_losses)}')
    scaler = 'no scaler for now...'
    return np.mean(val_losses), model, scaler


def create_base_network(params: dict) -> nn.Module:
    """
    Creates the base network. Two copies of this network are used to create the siamese twins.
    :param params: parameter dictionary
    :return:
    """

    n_convs = params['num_convs']
    activation_func = get_activation(params)
    channels = [1] + params['channels']
    layers = []
    for i in range(n_convs):
        layers.append(nn.BatchNorm2d(channels[i]))
        layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                kernel_size=params['kernel_size'][i], padding=1))
        layers.append(activation_func)
        layers.append(nn.MaxPool2d(kernel_size=2))
    layers.append(nn.Flatten(start_dim=1))
    model = nn.Sequential(*layers)
    return model