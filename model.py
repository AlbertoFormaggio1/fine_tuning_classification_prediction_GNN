import torch.nn as nn
from torch_geometric.nn import GCNConv

########## QUESTION: SHOULD DROPOUT BE ADDED?
########## https://dl.acm.org/doi/pdf/10.1145/3487553.3524725
class MLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: list[int], dropout: float = 0):
        super().__init__()
        layers = []

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features=input_size, out_features=h))
            layers.append(nn.ReLU()) # Can be substituted by something else?
            layers.append(nn.Dropout(p=dropout)) # Can be omitted? Or which is the best value for it?

        layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=num_classes))
        self.MLP = nn.Sequential(layers)

    def forward(self, x):
        return self.MLP(x)

# Check the parameters of GCN to find the best configuration.
class GCN(nn.Module):
    def __init__(self, input_size: int, hidden_channels: int, class_num: int, dropout: float = 0.5):
        super().__init__()
        # Should parameter improved = True?
        # Cached should be used for transductive learning, which is the case of our link prediction.
        # we need to see if it's possible to modify it when changing task or not
        self.conv1 = GCNConv(input_size, hidden_channels, improved=True)
        self.conv2 = GCNConv(hidden_channels, class_num, improved=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x




class GCN_MLP(nn.Module):
    def __init__(self, gcn, mlp):
        super().__init__()
        self.gcn = gcn
        self.mlp = mlp

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return self.mlp(x)

