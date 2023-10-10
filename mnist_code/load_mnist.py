import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class LoadMNIST():
    def __init__(self) -> None:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
        self.data_folder = os.path.join(PROJECT_ROOT, 'data/')

    def load_raw_mnist(self):
        transform = transforms.ToTensor()
        raw_data = os.path.join(self.data_folder, 'raw/')
        trainset = datasets.MNIST(root=raw_data,
                                  train=True, 
                                  download=True, 
                                  transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=64,
                                                   shuffle=True)

        testset = datasets.MNIST(root=raw_data,
                                 train=False,
                                 download=True,
                                 transform=transform)
        
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=64,
                                                  shuffle=True)
        
        return train_loader, test_loader

