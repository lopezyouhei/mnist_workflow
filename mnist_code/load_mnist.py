import os
import torchvision.datasets as datasets

class LoadMNIST():
    def __init__(self) -> None:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
        self.data_folder = os.path.join(PROJECT_ROOT, 'data/')

    def load_raw_mnist(self):
        raw_data = os.path.join(self.data_folder, 'raw/')
        trainset = datasets.MNIST(root=raw_data,
                                  train=True, 
                                  download=True, 
                                  transform=None)
        testset = datasets.MNIST(root=raw_data,
                                 train=False,
                                 download=True,
                                 transform=None)
        
        return trainset, testset

