import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch.functional import F

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
    def __init__(self, input_size: int, hidden_channels: int, embedding_size: int, dropout: float = 0.5):
        super().__init__()
        # Should parameter improved = True?
        # Cached should be used for transductive learning, which is the case of our link prediction.
        # we need to see if it's possible to modify it when changing task or not
        self.conv1 = GCNConv(input_size, hidden_channels, improved=True)
        self.conv2 = GCNConv(hidden_channels, embedding_size, improved=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):

    def __init__(self, input_size: int, embedding_size: int, hidden_channels: int = 16, heads:int = 8):
        super().__init__()
        # 256 channels seemed the best in the paper (but it depends on the complexity of the dataset)
        # LR = 0.001/0.01
        self.conv1 = GATv2Conv(input_size, hidden_channels, heads=heads)
        # Maybe concat should be set to False for the last layer so that the outputs will be averaged.
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels * heads, embedding_size, heads=1)

    def forward(self, x, edge_index):
        # They say dropout didn't change the results, but maybe since the model is powerful and our dataset is not, we may try something
        # to smooth the results
        x = F.dropout(x, 0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = nn.ELU()(x)
        x = F.dropout(x, 0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = nn.ELU()(x)
        x = F.dropout(x, 0.6, training=self.training)
        x = self.conv3(x, edge_index)
        return F.softmax(x, dim=1)

class GAT_MLP(nn.Module):
    def __init__(self, gat, mlp):
        super().__init__()
        self.gat = gat
        self.mlp = mlp

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return self.mlp(x)


class GCN_MLP(nn.Module):
    def __init__(self, gcn, mlp):
        super().__init__()
        self.gcn = gcn
        self.mlp = mlp

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return self.mlp(x)

