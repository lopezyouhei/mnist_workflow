import matplotlib.pyplot as plt
import math

def visualize_mnist(loader, num_images=8):

    dataiter = iter(loader)
    images, labels = dataiter.next()

    fig, axes = plt.subplots(nrows=2, 
                             ncols=math.ceil(num_images/2),
                             figsize=(10,2))
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        ax.title.set_text(str(labels[i].item()))
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

