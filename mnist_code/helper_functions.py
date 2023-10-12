import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_mnist(loader, num_images=8):

    dataiter = iter(loader)
    images, labels = next(dataiter)

    fig, axes = plt.subplots(nrows=2, 
                             ncols=math.ceil(num_images/2),
                             figsize=(10,2))
    
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        ax.title.set_text("Label:" + str(labels[i].item()))
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_statistics(loader):
    len_loader = len(loader)

    mean, var = 0.0, 0.0
    for images, _ in loader:
        mean += images.mean()
    mean /= len_loader

    for images, _ in loader:
        var += (images - mean).pow(2).mean()
    std = (var/len_loader)**0.5

    mean = mean.item()
    std = std.item()

    print("Tensor mean and standard deviation")
    print(f"Mean: {mean:.3f}, StDev: {std:.3f}")
    print("Images (8-bit) mean and standard deviation")
    print(f"Mean: {int(mean*255)}, StDev: {int(std*255)}")

def _get_class_distribution(**datasets):
    """For each dataset provided create a distribution list and return it as a 
    dictionary. The datasets should be provided as single word kwarg arguments:
    i.e. _get_class_distribution(train=train_set, test=test_set)
    Where 'train' is the name of the dataset and 
    'train_set' is the torchvision.datasets.mnist.MNIST object

    Best practice is to place the datasets in the following order 
    (if available): train, validation, test

    Returns:
        dict: class distribution dictionary, key=name, value=distribution
    """

    class_distribution = dict()
    for name, dataset in datasets.items():
        labels = np.array(dataset.targets)
        unique, count = np.unique(labels, return_counts=True)
        distribution = np.asarrays((unique, count)).T
        class_distribution[name] = distribution

    return class_distribution

