import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import Any
CUDA_LAUNCH_BLOCKING=1

class SiameseNN(nn.Module):
    def __init__(self, params, device):
        super(SiameseNN, self).__init__()
        # Defining the device
        self.device = device

        # Defining baseline network (encoding)
        self.base_network = create_base_network(params, device=self.device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass
        :param x1: first twin set
        :param x2: second twin set
        :return:
            - :param d_tanh: model output. An output closer to 1 means the model predicts
                     that the two images are of the same class.
        """

        # Feeding both sets of twins through the base network (encoder)
        h1 = self.base_network(x1)
        h2 = self.base_network(x2)

        # Calculating the L1 mean difference between to encodings
        delta_h = torch.abs(h1 - h2)
        dh_size = delta_h.size(1)
        d_logit = nn.Linear(dh_size, 1, device=delta_h.device)(delta_h)
        #d_logit = torch.mean(delta_h, dim=1).reshape(-1, 1)

        # Tanh (When d_logit -> 0, then d_tanh -> 1. When d_logit -> +inf, d_tanh -> 0)
        #d_tanh = 1 - nn.Tanh()(d_logit)
        #d_tanh = 1 - nn.Tanh()(torch.abs(d_logit))
        d_tanh = nn.Sigmoid()(d_logit)
        return d_tanh


def get_activation(params: dict) -> nn.Module:
    """
    Used to choose the activation function during the experiment
    :param params: experiment parameters
    :return: the activation function
    """

    activation_map = {
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(),
        'ELU': nn.ELU()
    }
    return activation_map[params['activation']]


def get_scheduler(optimizer, params):
    """
    Used to choose the scheduler during the experiment
    :param optimizer: the optimizer used in the training loop (needed to define the scheduler)
    :param params: experiment parameters
    :return: the scheduler
    """

    scheduler_map = {
        'linear': torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0,
                                                    total_iters=params['num_epochs']),
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs']),
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    }
    """
    'exp': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma']),
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params['patience']),
    'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['num_epochs']//params['num_steps'],
                                            gamma=1 - params['gamma'])
    """
    return scheduler_map[params['scheduler']]


def should_stop(loss: list, patience: int = 3) -> bool:
    """
    implementation of early stopping
    :param loss: loss function list
    :param patience: if the loss rises for `patience` losses, we stop
    :return: increasing: if true, we should stop
    """
    if len(loss) <= patience:
       return False

    # Truncating, so we only check the last `patience` elements
    recent_loss = torch.as_tensor(loss[np.maximum(0, len(loss) - patience):])

    # If the loss hasn't decreased in `patience` epochs, we want the model to stop training
    increasing = all(x<y for x, y in zip(recent_loss, recent_loss[1:]))
    return increasing


def train_model_nn(X_train_1: torch.Tensor, X_train_2: torch.Tensor, y_train: torch.Tensor,
                   X_val_1: torch.Tensor, X_val_2: torch.Tensor, y_val: torch.Tensor,
                   params: dict, device: torch.device, check_val: bool = True) -> tuple[float, Any, Any]:
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
    val_dataset = TensorDataset(X_val_1, X_val_2, y_val)

    # Defining model
    model = SiameseNN(params, device=device)

    # Defining training parameters
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = get_scheduler(optimizer, params)

    epoch_losses = []
    val_losses = []

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    # Training loop
    for epoch in range(params['num_epochs']):
        model.train()
        epoch_loss = 0
        curr_batch = 0
        # for each batch in the epoch
        for X_1_batch, X_2_batch, y_batch in train_loader:
            X_1_batch = X_1_batch.to(device)
            X_2_batch = X_2_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_1_batch, X_2_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_1_batch.size(0) / len(train_loader.dataset)
            curr_batch += 1
            #print(f'batch: {curr_batch} out of {len(train_loader)}. loss: {loss.item()}')
        epoch_losses.append(epoch_loss)

        # Checking validation loss for this epoch
        if check_val:
            val_loss_batches = []
            model.eval()
            for X_1_batch, X_2_batch, y_batch in val_loader:
                with torch.no_grad():
                    y_val_pred = model(X_1_batch.to(device), X_2_batch.to(device))
                    val_loss_batches.append(criterion(y_val_pred, y_batch.to(device)).cpu())
            val_loss = torch.mean(torch.as_tensor(val_loss_batches))
            val_losses.append(val_loss)
            # Early stopping
            if should_stop(val_losses):
                break
                print("huh")

        # print(len(val_losses))

        print(f'epoch: {epoch}, loss: {epoch_loss}, val_loss: {val_loss}')
    scaler = 'no scaler for now...'
    return val_losses, model, scaler


def create_base_network(params: dict, device: torch.device) -> nn.Module:
    """
    Creates the base network. Two copies of this network are used to create the siamese twins.
    :param params: parameter dictionary
    :return:
    """

    n_convs = params['num_convs']
    convs_per_block = params['convs_per_block']
    activation_func = get_activation(params)
    channels = [1] + params['channels']
    layers = []
    for i in range(n_convs):
        for j in range(convs_per_block):
            layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i],
                                    kernel_size=params['kernel_size'][i], padding=1))
            layers.append(activation_func)
            layers.append(nn.BatchNorm2d(channels[i]))
        layers.append(nn.MaxPool2d(kernel_size=2))
    layers.append(nn.Flatten(start_dim=1))
    model = nn.Sequential(*layers).to(device)
    return model