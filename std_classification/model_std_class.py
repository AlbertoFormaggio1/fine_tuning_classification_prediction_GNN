import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

"""
************************************ BASE CLASSES ************************************

We instantiate the models starting from the Link predictor, which is already providing the functions to compute the similarities 
between embeddings. This will be the same for every model, they will inherit this methods.
"""
class LinkPredictor(nn.Module):
    def decode(self, embeddings, edge_label_index):
        # Computing cosine similarity between embeddings in the training set + negative examples returned by the function for sampling
        simil = embeddings[edge_label_index[0]] * embeddings[edge_label_index[1]]
        return simil.sum(dim=-1)

    def decode_all(self, embedding):
        # Compute the similarity as ZZ^T
        prob_adj = embedding @ embedding.t()
        # Returns indices of nonzero elements. the traspose is so that the results will be as a row vector (im not sure, try to run it and check)
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, edge_index):
        # It is not possible to create instances of link predictor not associated with any architecture
        raise NotImplementedError


"""
************************************ MODEL CLASSES ************************************
"""
# Check the parameters of GCN to find the best configuration.
# https://arxiv.org/abs/1609.02907
class GCN(LinkPredictor):
    def __init__(self, input_size: int, embedding_size: int, hidden_channels: int = 16, dropout: float = 0.5):
        super().__init__()
        # Should parameter improved = True?
        # Cached should be used for transductive learning, which is the case of our link prediction.
        # we need to see if it's possible to modify it when changing task or not
        self.conv1 = GCNConv(input_size, hidden_channels, improved=True)
        self.conv2 = GCNConv(hidden_channels, embedding_size, improved=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.ELU()(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# https://arxiv.org/abs/2105.14491
class GAT(LinkPredictor):
    def __init__(self, input_size: int, embedding_size: int, hidden_channels: int = 16, heads: int = 8, heads_out: int = 1, dropout: float = 0.6):
        super().__init__()
        # 256 channels seemed the best in the paper (but it depends on the complexity of the dataset)
        # LR = 0.001/0.01
        self.conv1 = GATv2Conv(input_size, hidden_channels, heads=heads)
        # Maybe concat should be set to False for the last layer so that the outputs will be averaged.
        self.conv2 = GATv2Conv(hidden_channels * heads, embedding_size, heads=heads_out)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # They say dropout didn't change the results, but maybe since the model is powerful and our dataset is not, we may try something
        # to smooth the results
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = nn.ELU()(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# https://arxiv.org/pdf/1706.02216v4.pdf
class Graph_SAGE(LinkPredictor):
    def __init__(self, input_size: int, embedding_size: int, hidden_channels: int = 512, dropout: float = 0.5):
        # Using 2 layers has led to the best results in the original paper.
        # furthermore, they reported that max and LSTM were the best and similar in terms of accuracy.
        # However, pooling was a bit faster than LSTM.
        # We consider as default hidden_size the one for the "small" pool network
        super().__init__()
        self.sage1 = SAGEConv(input_size, hidden_channels, aggr='max')
        self.sage2 = SAGEConv(hidden_channels, embedding_size, aggr='max')
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = nn.ELU()(x)
        x = self.dropout(x)
        x = self.sage2(x, edge_index)
        return x
