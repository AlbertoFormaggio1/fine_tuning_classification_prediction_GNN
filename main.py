import os
import json

import torch
import torch_geometric.transforms as T

import engine
import load_dataset
import model
import utils
import parameters
import get_best_params

# select the device on which you should run the computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#************************************** COMMANDS ************************************

use_grid_search = True #False
dataset_name = "cora"   # cora - citeseer - pubmed
nets = ["GCN"]          # GCN - GAT - SAGE


# ************************************ PARAMETERS ************************************

#GCN
parameters_grid_GCN = parameters.parameters_grid_GCN
parameters_GCN = parameters.parameters_GCN

#GAT
parameters_grid_GAT = parameters.parameters_grid_GAT
parameters_GAT = parameters.parameters_GAT

# SAGE
parameters_grid_SAGE = parameters.parameters_grid_SAGE
parameters_SAGE = parameters.parameters_SAGE

# Others
lr = parameters.lr
weight_decay = parameters.weight_decay


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


# ************************************ TRAINING ************************************

for net in nets:

    results_file = os.path.join(dataset_name + "_" + net + "_results.json")
    if(os.path.exists(results_file)):
        with open(results_file) as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    params_file = os.path.join(dataset_name + "_" + net + "_params.json")
    if(os.path.exists(params_file)):
        with open(params_file) as f:
            params_dict = json.load(f)
    else:
        params_dict = {}

    if net == "GCN":
        if use_grid_search :
            param_combinations = utils.generate_combinations(parameters_grid_GCN)
        else :
            param_combinations = [parameters_GCN]
    elif net == "GAT":
        if use_grid_search :
            param_combinations = utils.generate_combinations(parameters_grid_GAT)
        else :
            param_combinations = [parameters_GAT]
    else:
        if use_grid_search :
            param_combinations = utils.generate_combinations(parameters_grid_SAGE)
        else :
            param_combinations = [parameters_SAGE]

    i = 1
    for params in param_combinations:

        print("\n " + net + ", (iteration " + str(i) + " over " + str(len(param_combinations)) + ") - Testing parameters: ")
        i += 1
        for key, value in params.items():
            print(f"{key}: {value}", end="\n") 
        print("--------------------------------\n")

        if net == "SAGE":
            batch_generation = True
            num_batch_neighbors = params["num_batch_neighbors"]
        else:
            batch_generation = False
            num_batch_neighbors = []

        # ************************************ CLASSIFICATION 1 ************************************

        input_size = classification_dataset.num_features
        hidden_channels = params["hidden_channels"]
        output_size = params["embedding_size"]
        dropout = params["dropout"]

        if net == "GCN":
            network = model.GCN(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels, dropout=dropout)
        elif net == "GAT":
            heads = params["heads"]
            network = model.GAT(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels, heads=heads, dropout=dropout)
        else:
            network = model.Graph_SAGE(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels, dropout=dropout)

        input_size_mlp = params["embedding_size"]
        output_size_mlp = classification_dataset.num_classes
        hidden_sizes_mlp = params["hidden_sizes_mlp_class1"]
        dropout_mlp = params["dropout_mlp_class1"]
        mlp_classification1 = model.MLP(input_size=input_size_mlp, num_classes=output_size_mlp, hidden_sizes=hidden_sizes_mlp, dropout=dropout_mlp)

        if net == "GCN":
            model_classification1 = model.GCN_MLP(network, mlp_classification1)
        elif net == "GAT":
            model_classification1 = model.GAT_MLP(network, mlp_classification1)
        else:
            model_classification1 = model.SAGE_MLP(network, mlp_classification1)
        

        # define the loss function and the optimizer. The learning rate is found on papers, same goes for the learning rate decay
        # and the weight decay
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion => CrossEntropyLoss in the case of classification
        optimizer = torch.optim.Adam(model_classification1.parameters(), lr=lr, weight_decay=weight_decay)

        # run the training
        epochs = params["epochs_classification1"]
        results_class1 = engine.train_classification(model_classification1, classification_dataset.data, classification_dataset.data, criterion,
                            optimizer, epochs, batch_generation, num_batch_neighbors)

        print("CLASSIFICATION 1 RESULTS")
        for k,v in results_class1.items():
            print(k + ":" + str(v[-1]))
        print()

        # ************************************ LINK PREDICTION ************************************

        input_size_mlp = params["embedding_size"]
        output_size_mlp = linkpred_dataset.num_classes  # Non è legato al numero di classi ## e allora che mettiamo ? 
        hidden_sizes_mlp = params["hidden_sizes_mlp_link_pred"]
        dropout_mlp = params["dropout_mlp_link_pred"] 
        mlp_linkpred = model.MLP(input_size=input_size_mlp, num_classes=output_size_mlp, hidden_sizes=hidden_sizes_mlp, dropout=dropout_mlp)

        if net == "GCN":
            model_linkpred = model.GCN_MLP(network, mlp_linkpred)
        elif net == "GAT":
            model_linkpred = model.GAT_MLP(network, mlp_linkpred)
        else:
            model_linkpred = model.SAGE_MLP(network, mlp_linkpred)
        

        criterion = torch.nn.BCEWithLogitsLoss()

        # run the training
        epochs_linkpred = params["epochs_linkpred"]
        net_freezed_linkpred = params["net_freezed_linkpred"]

        optimizer = torch.optim.Adam(mlp_linkpred.parameters(), lr=lr, weight_decay=weight_decay)
        epochs = int(epochs_linkpred*net_freezed_linkpred)
        engine.train_link_prediction(model_linkpred, train_ds, criterion, optimizer, epochs, batch_generation, num_batch_neighbors)

        optimizer = torch.optim.Adam(model_linkpred.parameters(), lr=lr, weight_decay=weight_decay)
        epochs = epochs_linkpred - epochs
        engine.train_link_prediction(model_linkpred, train_ds, criterion, optimizer, epochs, batch_generation, num_batch_neighbors)

        engine.eval_predictor(model_linkpred, criterion, val_ds, batch_generation)


        # ************************************ CLASSIFICATION 2 ************************************

        input_size_mlp = params["embedding_size"]
        output_size_mlp = classification_dataset.num_classes
        hidden_sizes_mlp = params["hidden_sizes_mlp_class2"]
        dropout_mlp = params["dropout_mlp_class2"]
        mlp_classification2 = model.MLP(input_size=input_size_mlp, num_classes=output_size_mlp, hidden_sizes=hidden_sizes_mlp, dropout=dropout_mlp)

        if net == "GCN":
            model_classification2 = model.GCN_MLP(network, mlp_classification2)
        elif net == "GAT":
            model_classification2 = model.GAT_MLP(network, mlp_classification2)
        else:
            model_classification2 = model.SAGE_MLP(network, mlp_classification2)

        criterion = torch.nn.CrossEntropyLoss()

        # run the training
        epochs_classification2 = params["epochs_classification2"]
        net_freezed_classification2 = params["net_freezed_classification2"]

        optimizer = torch.optim.Adam(mlp_classification2.parameters(), lr=lr, weight_decay=weight_decay)
        epochs = int(epochs_classification2*net_freezed_classification2)
        results_class2a = engine.train_classification(model_classification2, classification_dataset.data, classification_dataset.data, criterion,
                            optimizer, epochs, batch_generation, num_batch_neighbors)
        
        print("CLASSIFICATION 2a RESULTS")
        for k,v in results_class2a.items():
            print(k + ":" + str(v[-1]))
        print()

        optimizer = torch.optim.Adam(model_classification2.parameters(), lr=lr, weight_decay=weight_decay)
        epochs = epochs_classification2 - epochs
        results_class2b = engine.train_classification(model_classification2, classification_dataset.data, classification_dataset.data, criterion,
                            optimizer, epochs, batch_generation, num_batch_neighbors)

        print("CLASSIFICATION 2b RESULTS")
        for k,v in results_class2b.items():
            print(k + ":" + str(v[-1]))
        print()

        # ************************************ SAVING RESULTS ************************************

        # params_string = ""     # part of the key that explicit the parameters used
        # for k, v in params.items():
        #     params_string = params_string + "_" + k[0:3] + "_" + str(v)

        # Set key to use in dictionaries
        key = net + "||"
        for k, v in params.items():
            key = key + k[0:3] + "_" + str(v) + "/"

        # Save parameters used in the training
        params_list = []
        for k, r in params.items():
            params_list.append((k, r))
        params_dict[key] = params_list
        with open(params_file, "w") as f:
            json.dump(params_dict, f, indent = 4)

        # Save results of the training
        results_class1_list = []
        for k, r in results_class1.items():
            results_class1_list.append((k, r))

        results_class2a_list = []
        for k, r in results_class2a.items():
            results_class2a_list.append((k, r))

        results_class2b_list = []
        for k, r in results_class2b.items():
            results_class2b_list.append((k, r))

        results_dict[key] = [("results_class1", results_class1_list),
                             ("results_class2a", results_class2a_list),
                             ("results_class2b", results_class2b_list) ]
        
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent = 4)

    filename = dataset_name + "_" + net + "_best_runs.txt"
    get_best_params.find_best_params(dataset_name, net, results_dict, params_dict, 5, print_output=False, save_output=True, file_name=filename)