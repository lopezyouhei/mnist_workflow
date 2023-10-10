import torch
import torchvision.datasets as datasets

class LoadMNIST():
    def __init__(self) -> None:
        self.model = 2

    def load_raw_mnist(self):
        trainset = datasets.MNIST(root='./data/raw/',
                                  train=True, 
                                  download=True, 
                                  transform=None)
        testset = datasets.MNIST(root='./data/raw/',
                                 train=False,
                                 download=True,
                                 transform=None)
        
        return trainset, testset

