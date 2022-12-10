import torch
import torchvision.transforms as transforms
from torch.utils import data


class ImageDataset(data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, element, transform):
        # Initialization
        self.element = element
        self.transform = transform

    def __len__(self):
        # Denotes the total number of samples
        return len(self.element)

    def __getitem__(self, index):
        # Generates one sample of data
        # Load data and get label
        X, y = self.element[index]

        X = self.transform(X)

        # Avoids copying the data
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

        return X, y
