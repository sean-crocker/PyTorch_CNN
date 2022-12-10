from sklearn.metrics import classification_report, f1_score, accuracy_score
from torchvision import transforms

import model
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader

from train import load_dataset_elements

torch.manual_seed(42)
np.random.seed(42)


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
                plt.savefig('outputs/prediction{}.png'.format(images_so_far))
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = model.resnet().to(device)
# load the model checkpoint
checkpoint = torch.load('outputs/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# Add testing data here
data_ = load_dataset_elements("test_root/")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_dataset = ImageDataset(data_, transform=transform)

data_loader = DataLoader(image_dataset, batch_size=32, shuffle=True)

eval_model(device, model, data_loader)

visualize_model(device, model, data_loader, len(image_dataset))

