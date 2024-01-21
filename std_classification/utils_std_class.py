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

# takes as input a dictionary made of key:lists, and the output is a list made of dictionary with all possible combinations
# ex. input = {'A':[0,1] 'B':[2,3]} --> output = [{'A':0, 'B':2}, {'A':1, 'B':2}, {'A':0, 'B':3}, {'A':1, 'B':3}]
def generate_combinations(input_dict):
    
    keys = list(input_dict.keys())
    lists = [input_dict[key] for key in keys]
    counters = [0] * len(lists)
    finish = False
    combination_list = []

    while not finish:
        current_dict = {}
        for i in range(len(keys)):
            current_dict [keys[i]] = lists[i][counters[i]]

        combination_list.append(current_dict)
        j = 0

        while j < len(lists) and counters[j] == len(lists[j]) - 1:
            counters[j] = 0
            j += 1

        if j == len(lists):
            finish = True
        else:
            counters[j] += 1

    return combination_list
