from torchvision import models
import torch.nn as nn


def resnet():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)

    # model = models.densenet121(pretrained=True)
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, 3)

    # model = models.vgg16_bn(pretrained=True)
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ftrs, 3)

    # model = models.vgg16(pretrained=True)
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ftrs, 3)

    # model = models.squeezenet1_1(pretrained=True)
    # model.classifier[1] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
    # model.num_classes = 3

    # model = models.mobilenet_v3_small(pretrained=True)
    # num_ftrs = model.classifier[3].in_features
    # model.classifier[3] = nn.Linear(num_ftrs, 3, bias=True)

    return model

"""""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3 * 3 * 64, 10)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(10, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
        
"""

