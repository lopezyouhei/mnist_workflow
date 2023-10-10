import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class LoadMNIST():
    def __init__(self) -> None:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
        self.data_folder = os.path.join(PROJECT_ROOT, 'data/')

    def load_raw(self, transform=None):
        """The function downloads (if necessary) and returns the train
        and test MNIST datasets.

        Returns:
            _type_: _description_
        """
        raw_data = os.path.join(self.data_folder, 'raw/')
        trainset = datasets.MNIST(root=raw_data,
                                  train=True, 
                                  download=True, 
                                  transform=transform)
        

        testset = datasets.MNIST(root=raw_data,
                                 train=False,
                                 download=True,
                                 transform=transform)
        
        
        
        return trainset, testset
    
    def load_tensor(self, batch_size=32):

        train_set, test_set = self.load_raw(transform=transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  shuffle=True)
        
        return train_loader, test_loader


