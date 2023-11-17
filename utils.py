import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Function for graph visualization
def visualize(h, color):
    # Create a TSNE object to plot into 2 dimensions data
    # Take the values to plot, convert them into cpu object and finally to numpy
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    # Don't show anything along x and y axis
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=60, c=color, cmap='Set2')
    plt.show()