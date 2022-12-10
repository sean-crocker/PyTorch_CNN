import copy

import torch
import numpy as np
from sklearn.metrics import accuracy_score


def train(device, model, criterion, optimizer, dataloader):
    prediction_array = []
    real_values = []
    train_error_tmp = []
    counter = 0

    model.train()  # Set model to training mode

    for inputs, targets in dataloader:
        counter += 1
        inputs, targets = inputs.to(device), targets.float().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            prediction = torch.round(torch.sigmoid(outputs))

            train_error_tmp.append(loss.item())
            loss.backward()
            optimizer.step()

        prediction_array.extend(prediction.cpu().detach().numpy())
        real_values.extend(targets.cpu().detach().numpy())

    epoch_loss = np.mean(train_error_tmp)
    epoch_acc = accuracy_score(real_values, prediction_array)

    return epoch_acc, epoch_loss


def validate(device, model, criterion, dataloader, best_acc, best_model_wts):
    prediction_array = []
    real_values = []
    validation_error_tmp = []
    running_loss = 0

    model.eval()  # Set model to evaluate mode

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.float().to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            running_loss += loss.item()

            prediction = torch.round(torch.sigmoid(outputs))

            validation_error_tmp.append(loss.item())

        prediction_array.extend(prediction.cpu().detach().numpy())
        real_values.extend(targets.cpu().detach().numpy())

    epoch_loss = np.mean(validation_error_tmp)
    epoch_acc = accuracy_score(real_values, prediction_array)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    return epoch_acc, epoch_loss, best_acc, best_model_wts
