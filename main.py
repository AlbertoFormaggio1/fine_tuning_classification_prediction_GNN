import torch
import os
import load_dataset
import engine
import model
import torch_geometric.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classification = False

if classification:

    transform_classification = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device)
    ])

    datasets = {}

    datasets['cora'] = load_dataset.load_ds('Cora', transform_classification)
    datasets['citeseer'] = load_dataset.load_ds('CiteSeer', transform_classification)
    datasets['pubmed'] = load_dataset.load_ds('PubMed', transform_classification)

    for ds in datasets.values():
        load_dataset.print_ds_info(ds)
        print('\n#################################\n')

    dataset = datasets['cora']

    model = model.GAT(dataset.num_features, dataset.num_classes)

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion => CrossEntropyLoss in the case of classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    results = engine.train(model, dataset.data, dataset.data, criterion, optimizer, 10, False)

    for k, r in results.items():
        print(k, r)

else:
    transform_prediction = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False)
    ])

    datasets = {}

    datasets['cora'] = load_dataset.load_ds('Cora', transform_prediction)
    datasets['citeseer'] = load_dataset.load_ds('CiteSeer', transform_prediction)
    datasets['pubmed'] = load_dataset.load_ds('PubMed', transform_prediction)

    dataset = datasets['cora']
    train_ds, val_ds, test_ds = dataset[0]

    model = model.GCN_Predictor(dataset.num_features, dataset.num_classes)

    criterion = torch.nn.BCEWithLogitsLoss()  # Define loss criterion => Binary Cross Entropy for link prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    engine.train_link_prediction(model, train_ds, criterion, optimizer, 101)

    acc = engine.test(model, val_ds)
    print(acc)