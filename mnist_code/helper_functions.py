import matplotlib.pyplot as plt
import numpy as np
import math
import torch

def visualize_mnist(loader, num_images=8, 
                    binary=False, binary_threshold=0.5):
    """Print examples from a loader object with labels above each image.

    Args:
        loader (torch.utils.data.data.dataloader.DataLoader): torch loader 
        consisting of image, label pairs
        num_images (int, optional): Number of images to be displayed over 2 
        rows. Defaults to 8.
    """
    
    # convert the loader into an iterator
    dataiter = iter(loader)
    # get images, labels from iterator
    # length will be same as loader batch size
    images, labels = next(dataiter)

    # create subplot instance with 2 rows
    fig, axes = plt.subplots(nrows=2, 
                             ncols=math.ceil(num_images/2),
                             figsize=(10,2))

    # ravel the axes: [2,4] -> 8 plots and iterate through them
    for i, ax in enumerate(axes.ravel()):
        if binary:
            im =  images[i] > binary_threshold
        else:
            im = images[i]
        # show i-th image in greyscale
        ax.imshow(im.squeeze().numpy(), cmap='gray')
        # set i-th label as title for i-th image
        ax.title.set_text("Label:" + str(labels[i].item()))
        # turns off some plot features to improve image visualization
        ax.axis('off')
    # display plot in tight layout
    if binary:
        fig.suptitle(f"Binary - Threshold({binary_threshold})")
    else:
        fig.suptitle("Grayscale")

    plt.tight_layout()
    plt.show()

def _print_statistics(mean, stdev):
    if type(mean)==float and type(stdev)==float:
        # print tensor and image statistics
        print("Tensor mean and standard deviation")
        print(f"Mean: {mean:.3f}, StDev: {stdev:.3f}")
        # convert mean and std to 8-bit format
        print("Images (8-bit) mean and standard deviation")
        print(f"Mean: {int(mean*255)}, StDev: {int(stdev*255)}")
    
    elif type(mean)==dict and type(stdev)==dict:
        for i in range(len(mean)):
            class_mean = mean[i]
            class_stdev = stdev[i]
            # print tensor and image statistics
            print(f"Label {i}:\n")
            print("Tensor mean and standard deviation")
            print(f"Mean: {class_mean:.3f}, StDev: {class_stdev:.3f}")
            # convert mean and std to 8-bit format
            print("Images (8-bit) mean and standard deviation")
            print(f"Mean: {int(class_mean*255)}, StDev: {int(class_stdev*255)}\n")

def get_statistics(loader):
    """Prints the mean and standard deviation of data loader. It's expected 
    that the loader is only defined with ToTensor() transformation and that the
    data are single channel images.

    Args:
        loader (torch.utils.data.data.dataloader.DataLoader): torch dataloader
    """
    # length of dataloader -> number of samples in this case
    len_loader = len(loader)
    # initialize mean and var
    mean, var = 0.0, 0.0
    # sum up all images means
    for images, _ in loader:
        mean += images.mean()
    # divide by length of dataloader to get mean
    mean /= len_loader
    # sum up all images variances, since we subtract mean for each pixel in the
    # image we have to average all the local variances with ".mean()" to get 
    # average variance within the image
    for images, _ in loader:
        var += (images - mean).pow(2).mean()
    # calculate standard deviation
    stdev = (var/len_loader)**0.5

    # get mean and std value from torch.tensor
    mean = mean.item()
    stdev = stdev.item()
    _print_statistics(mean, stdev)
    

def get_class_statistics(loader):
    
    class_means = {i: 0.0 for i in range(10)}
    class_vars = {i: 0.0 for i in range(10)}
    class_counts = {i: 0 for i in range(10)}
    
    for images, labels in loader:
        for i in range(10):
            class_images = images[labels == i]
            if class_images.nelement() > 0:
                class_means[i] += class_images.mean().item()
                class_counts[i] += 1
    
    for i in range(10):
        class_means[i] /= class_counts[i]
    
    # Calculate variance for each class
    for images, labels in loader:
        for i in range(10):
            class_images = images[labels == i]
            if class_images.nelement() > 0:
                class_vars[i] += ((class_images - class_means[i]).pow(2).mean().item())

    for i in range(10):
        class_vars[i] /= class_counts[i]

    # Calculate standard deviation for each class
    class_stdevs = {i: class_vars[i]**0.5 for i in range(10)}

    _print_statistics(class_means, class_stdevs)

def _get_class_distribution(**datasets):
    """For each dataset provided create a distribution list and return it as a 
    dictionary. The datasets should be provided as single word kwarg arguments
    or dictionary:
    i.e. _get_class_distribution(train=train_set, test=test_set)
    Where 'train' is the name of the dataset and 
    'train_set' is the torchvision.datasets.mnist.MNIST object

    Best practice is to place the datasets in the following order 
    (if available): train, validation, test

    Returns:
        dict: class distribution dictionary, key=name, value=distribution
    """
    # instantiate class_distribution dictionary
    class_distribution = dict()
    # iterate through datasets
    for name, dataset in datasets.items():
        # create array of labels of all items in dataset
        # length of array is length of dataset
        labels = np.array(dataset.targets)
        # order the labels and count number of instances for each label
        # unique = label, count = # of each label
        unique, count = np.unique(labels, return_counts=True)
        # combine unique classes and count of the class as an array
        distribution = np.asarray((unique, count)).T
        # store dataset name and dataset distribution in dictionary
        class_distribution[name] = distribution

    return class_distribution

def plot_distribution(**datasets):
    """For each dataset provided, plot the distributions side-by-side. The 
    datasets should be provided as single word kwarg arguments or dictionary: 
    i.e. plot_distribution(train=train_set, test=test_set)
    Where 'train' is the name of the dataset and '
    train_set' is the torchvision.datasets.mnist.MNIST object

    Best practice is to place the datasets in the following order 
    (if available): train, validation, test
    """

    # get dictionary of distributions for each dataset
    distribution_dict = _get_class_distribution(**datasets)
    # define bar_width depending on number of datasets user has inserted
    # rounds to 1 decimal, 0.8 is matplotlib default bar width 
    bar_width = round(0.8 / len(distribution_dict), 1)
    # get number of labels in dictionary value
    num_classes = len(next(iter(distribution_dict.values())))

    # create array of length num_classes for defining where each bar will be 
    # placed
    X_pos_init = np.arange(num_classes)
    # iterate through dataset in dictionary with a running number i
    for i, (dataset, distribution) in enumerate(distribution_dict.items()):
        # define bar position depending on which dataset
        x_pos = [x + (i*bar_width) for x in X_pos_init]
        # plot the bar
        plt.bar(x_pos, distribution[:, 1], width=bar_width, label=dataset)
    
    plt.xlabel('Class Label')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classes in MNIST Training Set')
    plt.xticks(np.arange(num_classes))
    plt.legend()
    plt.show()

# TODO: implement down-projection methods (e.g. PCA, t-SNE) to explore data