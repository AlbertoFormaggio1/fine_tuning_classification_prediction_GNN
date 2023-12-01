import os

import torch
import torch_geometric.transforms as T

import engine
import load_dataset
import model

# select the device on which you should run the computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ************************************ PARAMETERS ************************************

#net_name = "GAT"
dataset_name = "cora"

#GCN
parameters_GCN = {
    "embedding_size": 64,
    "hidden_channels" : 16,
    "dropout": 0.2
}

#GAT
parameters_GAT = {
    "embedding_size": 64,
    "hidden_channels" : 16,
    "heads" : 8,
    "dropout" : 0.6
}

# SAGE
parameters_SAGE = {
    "embedding_size": 64,
    "hidden_size" : 512,
    "dropout": 0.2
}

epochs_classification1 = 10
epochs_linkpred = 10
net_freezed_linkpred = 0.5
epochs_classification2 = 100
net_freezed_classification2 = 0.5

# aggiungere anche lr e decay rate e hidden_sizes?
# diversificare i parametri di rete/mlp tra class1/linkpred/class2 ?

results_file = os.path.join(dataset_name + "_results.json")
import json
if(os.path.exists(results_file)):
    with open(results_file) as f:
        final_dict = json.load(f)
else:
    final_dict = {}


# ************************************ CLASSIFICATION DATASET ************************************

# Normalize the features and put it on the appropriate device
transform_classification = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device)
])

classification_datasets = {}

# Load the 3 datasets and apply the transform needed
classification_datasets['cora'] = load_dataset.load_ds('Cora', transform_classification)
classification_datasets['citeseer'] = load_dataset.load_ds('CiteSeer', transform_classification)
classification_datasets['pubmed'] = load_dataset.load_ds('PubMed', transform_classification)

# print the information for each dataset
for ds in classification_datasets.values():
    load_dataset.print_ds_info(ds)
    print('\n#################################\n')

classification_dataset = classification_datasets[dataset_name]


# ************************************ LINK PREDICTION DATASET ************************************

# Change transform for link prediction
transform_prediction = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=False)
])

linkpred_datasets = {}

# Load the datasets as before
linkpred_datasets['cora'] = load_dataset.load_ds('Cora', transform_prediction)
linkpred_datasets['citeseer'] = load_dataset.load_ds('CiteSeer', transform_prediction)
linkpred_datasets['pubmed'] = load_dataset.load_ds('PubMed', transform_prediction)

linkpred_dataset = linkpred_datasets['cora']
# Get the 3 splits
train_ds, val_ds, test_ds = linkpred_dataset[0]

for net in ["SAGE"]:

    if net == "GCN":
        parameters = parameters_GCN
    elif net == "GAT":
        parameters = parameters_GAT
    else:
        parameters = parameters_SAGE

    batch_generation = net == "SAGE"

    # ************************************ CLASSIFICATION 1 ************************************

    input_size = classification_dataset.num_features
    output_size = parameters["embedding_size"]
    dropout = parameters["dropout"]

    if net == "GCN":
        hidden_channels = parameters["hidden_channels"]
        network = model.GCN(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels, dropout=dropout)
    elif net == "GAT":
        hidden_channels = parameters["hidden_channels"]
        heads = parameters["heads"]
        network = model.GAT(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels, heads=heads, dropout=dropout)
    else:
        hidden_size = parameters["hidden_size"]
        network = model.Graph_SAGE(input_size=input_size, embedding_size=output_size, hidden_size=hidden_size, dropout=dropout)
    
    input_size = parameters["embedding_size"]
    output_size = classification_dataset.num_classes
    hidden_sizes = [20]
    dropout = 0 # droupout forse meglio toglierlo nel MLP
    mlp_classification1 = model.MLP(input_size=input_size, num_classes=output_size, hidden_sizes=hidden_sizes, dropout=dropout)

    if net == "GCN":
        model_classification1 = model.GCN_MLP(network, mlp_classification1)
    elif net == "GAT":
        model_classification1 = model.GAT_MLP(network, mlp_classification1)
    else:
        model_classification1 = model.SAGE_MLP(network, mlp_classification1)
    

    # define the loss function and the optimizer. The learning rate is found on papers, same goes for the learning rate decay
    # and the weight decay
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion => CrossEntropyLoss in the case of classification
    optimizer = torch.optim.Adam(model_classification1.parameters(), lr=0.01, weight_decay=5e-4)

    # run the training
    epochs = epochs_classification1
    results = engine.train(model_classification1, classification_dataset.data, classification_dataset.data, criterion,
                           optimizer, epochs, batch_generation)


    # ************************************ LINK PREDICTION ************************************

    input_size = parameters["embedding_size"]
    output_size = linkpred_dataset.num_classes
    hidden_sizes = [5]
    dropout = 0 # droupout forse meglio toglierlo negli MLP
    mlp_linkpred = model.MLP(input_size=input_size, num_classes=output_size, hidden_sizes=hidden_sizes, dropout=dropout)

    if net == "GCN":
        model_linkpred = model.GCN_MLP(network, mlp_linkpred)
    elif net == "GAT":
        model_linkpred = model.GAT_MLP(network, mlp_linkpred)
    else:
        model_linkpred = model.SAGE_MLP(network, mlp_linkpred)
    

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(mlp_linkpred.parameters(), lr=0.01, weight_decay=5e-4)
    epochs = int(epochs_linkpred*net_freezed_linkpred)
    engine.train_link_prediction(model_linkpred, train_ds, criterion, optimizer, epochs, batch_generation)

    optimizer = torch.optim.Adam(model_linkpred.parameters(), lr=0.01, weight_decay=5e-4)
    epochs = epochs_linkpred - epochs
    engine.train_link_prediction(model_linkpred, train_ds, criterion, optimizer, epochs, batch_generation)


    # ************************************ CLASSIFICATION 2 ************************************

    input_size = parameters["embedding_size"]
    output_size = classification_dataset.num_classes
    hidden_sizes = [5]
    dropout = 0 # droupout forse meglio toglierlo negli MLP
    mlp_classification2 = model.MLP(input_size=input_size, num_classes=output_size, hidden_sizes=hidden_sizes, dropout=dropout)

    if net == "GCN":
        model_classification2 = model.GCN_MLP(network, mlp_classification2)
    elif net == "GAT":
        model_classification2 = model.GAT_MLP(network, mlp_classification2)
    else:
        model_classification2 = model.SAGE_MLP(network, mlp_classification2)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(mlp_classification2.parameters(), lr=0.01, weight_decay=5e-4)
    epochs = int(epochs_classification2*net_freezed_classification2)
    results = engine.train(model_classification2, classification_dataset.data, classification_dataset.data, criterion,
                           optimizer, epochs, batch_generation)

    optimizer = torch.optim.Adam(model_linkpred.parameters(), lr=0.01, weight_decay=5e-4)
    epochs = epochs_classification2 - epochs
    results = engine.train(model_classification2, classification_dataset.data, classification_dataset.data, criterion,
                           optimizer, epochs, batch_generation)
    

    # ************************************ SAVING RESULTS ************************************

    params = ""     # part of the key that explicit the parameters used
    for k, v in parameters.items():
        params = params + "_" + k[0:3] + "_" + str(v)

    for k, r in results.items():
        key = net + "_" + k + "_/" + params
        final_dict[key] = r

    with open(results_file, "w") as f:
        json.dump(final_dict, f, indent = 4)




#GCN
# parameters = [("embedding_size", 5), 
#     ("hidden_channels", 16), 
#     ("heads", 8),
#     ("dropout", 0.6)]

#GAT
# parameters = [("embedding_size", 5), 
#     ("hidden_channels", 16), 
#     ("dropout", 0.5)]

# SAGE
# parameters = [("embedding_size", 5), 
#     ("hidden_channels", 512), 
#     ("dropout", 0.5)]