import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_mnist(loader, num_images=8):
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
        # show i-th image in greyscale
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        # set i-th label as title for i-th image
        ax.title.set_text("Label:" + str(labels[i].item()))
        # turns off some plot features to improve image visualization
        ax.axis('off')
    # display plot in tight layout
    plt.tight_layout()
    plt.show()

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
    std = (var/len_loader)**0.5

    # get mean and std value from torch.tensor
    mean = mean.item()
    std = std.item()

    # print tensor and image statistics
    print("Tensor mean and standard deviation")
    print(f"Mean: {mean:.3f}, StDev: {std:.3f}")
    # convert mean and std to 8-bit format
    print("Images (8-bit) mean and standard deviation")
    print(f"Mean: {int(mean*255)}, StDev: {int(std*255)}")

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

    distribution_dict = _get_class_distribution(**datasets)
    bar_width = round(0.8 / len(distribution_dict), 1)

    num_classes = len(next(iter(distribution_dict.values())))

    X_pos_init = np.arange(num_classes)
    for i, (dataset, distribution) in enumerate(distribution_dict.items()):
        x_pos = [x + (i*bar_width) for x in X_pos_init]
        plt.bar(x_pos, distribution[:, 1], width=bar_width, label=dataset)
    
    plt.xlabel('Class Label')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classes in MNIST Training Set')
    plt.xticks(np.arange(num_classes))
    plt.legend()
    plt.show()
