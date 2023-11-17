import torch
import os
from torch_geometric.transforms import NormalizeFeatures
import load_dataset
import engine

datasets = {}

datasets['cora'] = load_dataset.load_ds('Cora', NormalizeFeatures())
datasets['citeseer'] = load_dataset.load_ds('CiteSeer', NormalizeFeatures())
datasets['pubmed'] = load_dataset.load_ds('PubMed', NormalizeFeatures())

for ds in datasets.values():
    load_dataset.print_ds_info(ds)
    print('\n#################################\n')

dataset = datasets['cora']

import torch
from torch.nn import Linear
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model)


criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

results = engine.train(model, dataset.data, dataset.data, criterion, optimizer, 50)

for r in results.values():
    print(r)