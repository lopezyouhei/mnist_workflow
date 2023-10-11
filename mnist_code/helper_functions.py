import matplotlib.pyplot as plt
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


    for key, value in datasets.items():
        