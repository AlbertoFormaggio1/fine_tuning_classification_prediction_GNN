import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch.functional import F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

########## QUESTION: SHOULD DROPOUT BE ADDED?
########## https://dl.acm.org/doi/pdf/10.1145/3487553.3524725
class MLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: list[int], dropout: float = 0):
        super().__init__()
        layers = []

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features=input_size, out_features=h))
            layers.append(nn.ELU()) # ELU has proved to be better than ReLU
            layers.append(nn.Dropout(p=dropout)) # Can be omitted? Or which is the best value for it?

        layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=num_classes))
        self.MLP = nn.Sequential(layers)

    def forward(self, x):
        return self.MLP(x)

# Check the parameters of GCN to find the best configuration.
# https://arxiv.org/abs/1609.02907
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

# https://arxiv.org/abs/2105.14491
class GAT(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_channels: int = 16, heads:int = 8):
        super().__init__()
        # 256 channels seemed the best in the paper (but it depends on the complexity of the dataset)
        # LR = 0.001/0.01
        self.conv1 = GATv2Conv(input_size, hidden_channels, heads=heads)
        # Maybe concat should be set to False for the last layer so that the outputs will be averaged.
        self.conv2 = GATv2Conv(hidden_channels * heads, embedding_size, heads=1)

    def forward(self, x, edge_index):
        # They say dropout didn't change the results, but maybe since the model is powerful and our dataset is not, we may try something
        # to smooth the results
        x = F.dropout(x, 0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = nn.ELU()(x)
        x = F.dropout(x, 0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# https://arxiv.org/pdf/1706.02216v4.pdf
class Graph_SAGE(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int, hidden_size: int = 512, dropout: float = 0.5):
        # Using 2 layers has led to the best results in the original paper.
        # furthermore, they reported that max and LSTM were the best and similar in terms of accuracy.
        # However, pooling was a bit faster than LSTM.
        # We consider as default hidden_size the one for the "small" pool network
        super().__init__()
        self.sage1 = SAGEConv(input_size, hidden_size, aggr='max')
        self.sage2 = SAGEConv(hidden_size, embedding_dim, aggr='max')
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = nn.ELU()(x)
        x = self.dropout(x)
        x = self.sage2(x, edge_index)
        return x

class SAGE_MLP(nn.Module):
    def __init__(self, sage, mlp):
        super().__init__()
        self.sage = sage
        self.mlp = mlp

    def forward(self, x):
        super().__init__()
        x = self.sage(x)
        x = self.mlp(x)
        return x

class GAT_MLP(nn.Module):
    """
    Please note that the class returns logits. They should be processed according to the graph task (e.g. softmax for
    node classification or sigmoid for link prediction).
    """
    def __init__(self, gat, mlp):
        super().__init__()
        self.gat = gat
        self.mlp = mlp

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.mlp(x)
        return x


class GCN_MLP(nn.Module):
    """
        Please note that the class returns logits. They should be processed according to the graph task (e.g. softmax for
        node classification or sigmoid for link prediction).
    """
    def __init__(self, gcn, mlp):
        super().__init__()
        self.gcn = gcn
        self.mlp = mlp

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return self.mlp(x)

