import copy
import glob
import json
import os

import numpy as np
import numpy.random
import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms

from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)


def load_dataset_elements(dataset_root):
    elements = []
    for image in glob.glob(os.path.join(dataset_root, "images", "image*.jpeg")):
        image_name = os.path.basename(image)
        labels_name = image_name.replace("image", "labels")
        labels_name = "{}.json".format(os.path.splitext(labels_name)[0])
        with open(os.path.join(dataset_root, "labels", labels_name)) as labels_file:
            # Gets associated label with image
            raw_labels = json.load(labels_file)
            labels = []
            for label in ("plastic bags", "plastic bottles", "other plastic"):
                labels.append(1 if raw_labels[label] else 0)
        elements.append((Image.open(image), labels))
    return elements


def plt_error(train_error, valid_error):
    plt.plot(train_error, '--', color="#111111", label='training error')
    plt.plot(valid_error, color="#111111", label='validation error')
    plt.xlabel("No of Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig('outputs/loss.png')
    plt.show()


def plt_accuracy(train_accuracy, valid_accuracy):
    plt.plot(train_accuracy, '--', color="#111111", label="training accuracy")
    plt.plot(valid_accuracy, color="#111111", label="validation accuracy")
    plt.xlabel("No of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig('outputs/acc.png')
    plt.show()


def get_data():
    data_ = load_dataset_elements("dataset_root/")
    train_data, val_data = train_test_split(data_, test_size=0.2)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {'train': ImageDataset(train_data, transform=data_transforms['train']),
                      'val': ImageDataset(val_data, transform=data_transforms['val'])}

    data_loaders = {'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
                    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=True)}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return data_loaders, dataset_sizes


def get_model(device):
    import model
    # initialize the model
    # model_ = model.CNN().to(device)
    model_ = model.resnet().to(device)
    # learning parameters
    optimizer = optim.AdamW(model_.parameters(), lr=1e-4)
    # optimizer = optim.AdamW(params=model_.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)
    criterion = nn.BCEWithLogitsLoss()

    return model_, optimizer, criterion


def save_model(model_, num_epochs, optimizer, criterion):
    # Save the trained model to disk
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion
    },  'outputs/model.pth')


def start(device, model_, criterion, optimizer, data_loaders, num_epochs=25):
    # start the training and validation
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model_.state_dict())
    for _ in tqdm.trange(num_epochs):
        train_epoch_acc, train_epoch_loss = train(device,
                                                  model_,
                                                  criterion,
                                                  optimizer,
                                                  data_loaders['train'])
        valid_epoch_acc, valid_epoch_loss, best_acc, best_model_wts = validate(device,
                                                                               model_,
                                                                               criterion,
                                                                               data_loaders['val'],
                                                                               best_acc,
                                                                               best_model_wts)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

    model_.load_state_dict(best_model_wts)
    return model_, train_loss, valid_loss, train_acc, valid_acc


def imshow(inp, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    inp = inp.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    if title is not None:
        ax.set_title(title)


def print_labels(labels):
    result = ""
    if labels[0] == 1:
        result += "Plastic Bag "
    if labels[1] == 1:
        result += "Plastic Bottles "
    if labels[2] == 1:
        result += "Other Plastic "
    if result == "":
        return "Nothing"
    return result


def visualize_model(device_, model_, dataloader, num_images=4):
    was_training = model_.training
    model_.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device_), labels.float().to(device_)
            outputs = model_(inputs)
            predicted = torch.round(torch.sigmoid(outputs))

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')

                title = "Predicted: {}\nActual: {}".format(print_labels(predicted[j]), print_labels(labels[j]))
                imshow(inputs.cpu().data[j], title=title)
                plt.pause(0.001)

                if images_so_far == num_images:
                    model_.train(mode=was_training)
                    return

        model_.train(mode=was_training)

    fig.set_size_inches(18.5, 10.5, forward=True)


def eval_model(device_, model_, dataloader):
    prediction_array = []
    real_values = []
    model_.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device_), labels.float().to(device_)
            outputs = model_(images)
            prediction = torch.round(torch.sigmoid(outputs))

            prediction_array.extend(prediction.cpu().detach().numpy())
            real_values.extend(labels.cpu().detach().numpy())

        plot_labels = ['plastic bags', 'plastic bottles', 'other plastic']
        print('Subset accuracy: {0}'.format(accuracy_score(real_values, prediction_array, normalize=True,
                                                           sample_weight=None)))
        print("Classification report: \n", (classification_report(real_values, prediction_array,
                                                                  target_names=plot_labels, zero_division=0)))
        print("F1 micro averaging:", (f1_score(real_values, prediction_array, average='micro')))


def main():
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 25

    model_ft, optimizer, criterion = get_model(device)

    data_loaders, dataset_sizes = get_data()

    model_ft, train_loss, valid_loss, train_acc, valid_acc = start(device, model_ft, criterion, optimizer, data_loaders,
                                                                   num_epochs)
    save_model(model_ft, num_epochs, optimizer, criterion)

    # Plot and save the train and validation line graphs
    plt_error(train_loss, valid_loss)
    plt_accuracy(train_acc, valid_acc)

    eval_model(device, model_ft, data_loaders['val'])

    visualize_model(device, model_ft, data_loaders['val'], num_images=dataset_sizes['val'])


if __name__ == '__main__':
    main()
