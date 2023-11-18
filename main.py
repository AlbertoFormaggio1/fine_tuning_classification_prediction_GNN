import torch
import os
from torch_geometric.transforms import NormalizeFeatures
import load_dataset
import engine
import model

datasets = {}

datasets['cora'] = load_dataset.load_ds('Cora', NormalizeFeatures())
datasets['citeseer'] = load_dataset.load_ds('CiteSeer', NormalizeFeatures())
datasets['pubmed'] = load_dataset.load_ds('PubMed', NormalizeFeatures())

for ds in datasets.values():
    load_dataset.print_ds_info(ds)
    print('\n#################################\n')

dataset = datasets['cora']

model = model.GAT(dataset.num_features, dataset.num_classes)

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion => CrossEntropyLoss in the case of classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

results = engine.train(model, dataset.data, dataset.data, criterion, optimizer, 50)

for r in results.values():
    print(r)