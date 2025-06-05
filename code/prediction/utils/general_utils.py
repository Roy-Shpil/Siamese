import pandas as pd
import torch
import torchvision
from typing import Any
from sklearn.model_selection import KFold
from utils.nn_utils import train_model_nn
import numpy as np
import os

import optuna


def load_files(processed_data_dir: str) -> dict:
    # processed_data_dir = '../../../data/set_pairs/processed_data'
    processed_data = {}
    for set in ['train', 'test']:
        for index in [1, 2]:
            for filetype in ['filenames', 'classes']:
                with open(f'{processed_data_dir}/{set}/{filetype}_{index}.txt', 'r', newline='') as file:
                    processed_data[f'{set}_{filetype}_{index}'] = pd.read_csv(file, sep=',').values.squeeze().tolist()

        with open(f'{processed_data_dir}/{set}/labels.txt', 'r', newline='') as file:
            processed_data[f'{set}_labels'] = pd.read_csv(file, sep=',').values.squeeze().tolist()
    return processed_data


def load_data(processed_data: dict, full_data_dir: str, set: str) -> torch.Tensor:
    sets = [[], []]
    for index in [1, 2]:
        for i, file in enumerate(processed_data[f'{set}_filenames_{index}']):
            filepath = f'{full_data_dir}/{processed_data[f"{set}_classes_{index}"][i]}/{file}.jpg'
            sets[index - 1].append(
                torchvision.io.read_image(path=filepath, mode=torchvision.io.ImageReadMode.GRAY) / 255.0)

    set_1 = torch.stack(sets[0], dim=0)
    set_2 = torch.stack(sets[1], dim=0)
    return set_1, set_2


def make_objective(X_train_val_1: torch.Tensor, X_train_val_2: torch.Tensor, y_train_val: torch.Tensor,
                   param_space: dict, device: torch.device, k: int = 5, print_message: bool = True):
    def objective(trial):
        params = suggest_params(trial, param_space)
        val_loss, models_kfold, scalers_kfold = k_fold_cross_val(X_train_val_1, X_train_val_2, y_train_val, params,
                                                                 device, k,
                                                                 print_message)
        trial.set_user_attr("model", models_kfold)
        trial.set_user_attr("scaler", scalers_kfold)
        return val_loss

    return objective


def suggest_params(trial, param_space):
    suggest_fns = {
        "int": trial.suggest_int,
        "float": trial.suggest_float,
        "categorical": trial.suggest_categorical,
    }

    params = {}
    for name, spec in param_space.items():
        kind = spec["suggestion"]
        if kind not in suggest_fns:
            raise ValueError(f"Unknown suggestion type: {kind}")

        suggest_fn = suggest_fns[kind]

        # Pass kwargs dynamically
        kwargs = {k: v for k, v in spec.items() if (k != "suggestion" and k != "listed")}
        if spec["listed"] is None:
            params[name] = suggest_fn(name, **kwargs)
        else:
            params[name] = [suggest_fn(f'{name}_{i}', **kwargs) for i in range(params[spec["listed"]])]

    return params


def k_fold_cross_val(X_train_val_1: torch.Tensor, X_train_val_2: torch.Tensor, y_train_val: torch.Tensor, params: dict,
                     device: torch.device, k:
        int = 5, print_message: bool = True) -> tuple[Any, list, list]:
    """
    trains the model using Kfold cross validation
    :param algorithm: algorithm of choosing (nn or xgboost)
    :param X_train_val: train+val features
    :param y_train_val: train+val outputs
    :param params: model paramerers
    :param k: number of folds
    :param print_message: if True, prints progress messages
    :return:
    """

    # Splitting the set to K folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    k_val_losses = []
    k_fold_models = []
    scalers = []
    i = 0
    print(f"\nParameters: {params}\n\n")

    # Training the model for each fold
    for train_idx, val_idx in kf.split(y_train_val):
        if print_message:
            print(f'training fold no. {i}')
        X_train_1, X_val_1 = X_train_val_1[train_idx, :, :, :], X_train_val_1[val_idx, :, :, :]
        X_train_2, X_val_2 = X_train_val_2[train_idx, :, :, :], X_train_val_2[val_idx, :, :, :]
        y_train, y_val = y_train_val[train_idx, :], y_train_val[val_idx, :]
        val_loss, model, scaler = train_model_nn(X_train_1, X_train_2, y_train, X_val_1, X_val_2, y_val, params, device,
                                                 check_val=True)
        k_val_losses.append(val_loss)
        k_fold_models.append(model)
        scalers.append(scaler)
        i += 1

    if print_message:
        # print(mean_k_loss.dtype)
        print(f"KFold complete! Final Validation Loss: {mean_k_loss}")
    return mean_k_loss.cpu(), k_fold_models, scalers


def study_best_params(X_train_val_1: torch.Tensor, X_train_val_2: torch.Tensor, y_train_val: torch.Tensor,
                      param_space: dict, device: torch.device, *, k: int = 5, print_message: bool = True, num_iters=30):
    study = optuna.create_study(direction="minimize")
    objective = make_objective(X_train_val_1, X_train_val_2, y_train_val, param_space, device, k, print_message)
    study.optimize(objective, n_trials=num_iters, callbacks=[save_best_model])

    print("Best hyperparameters:", study.best_params)

    models = study.user_attrs["best_model"]
    scalers = study.user_attrs["best_scaler"]
    scores = study.best_value

    return models, scalers, scores


def save_best_model(study: Any, trial: Any) -> None:
    """
    Saves the best model
    :param study: study (parameter search)
    :param trial: trial (any single instance of the parameter search)
    :return:
    """
    if study.best_value == trial.value:
        study.set_user_attr("best_model", trial.user_attrs["model"])
        study.set_user_attr("best_scaler", trial.user_attrs["scaler"])
