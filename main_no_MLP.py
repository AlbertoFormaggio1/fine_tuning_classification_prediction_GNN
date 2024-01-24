import datetime
import json
import os
import re

import torch
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard
from torch_geometric.nn import summary

import engine
import get_best_params
import load_dataset
import model
import parameters
import utils

random_seed = 42
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)

# select the device on which you should run the computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ************************************** COMMANDS ************************************

use_grid_search = True  # False
dataset_name = "pubmed"  # cora - citeseer - pubmed
nets = ["GAT"]  # GCN - GAT - SAGE

# ************************************ PARAMETERS ************************************

# GCN
parameters_grid_GCN = parameters.parameters_grid_GCN
parameters_GCN = parameters.parameters_GCN

# GAT
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
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False)
])

linkpred_datasets = {}

# Load the datasets as before
linkpred_datasets['cora'] = load_dataset.load_ds('Cora', transform_prediction)
linkpred_datasets['citeseer'] = load_dataset.load_ds('CiteSeer', transform_prediction)
linkpred_datasets['pubmed'] = load_dataset.load_ds('PubMed', transform_prediction)

linkpred_dataset = linkpred_datasets[dataset_name]
# Get the 3 splits
train_ds, val_ds, test_ds = linkpred_dataset[0]

# ************************************ TRAINING ************************************

for net in nets:

    out_dir = "nomlp_" + dataset_name + "_" + net
    os.makedirs(out_dir, exist_ok=True)

    results_file = os.path.join(out_dir, dataset_name + "_" + net + "_results.json")
    if (os.path.exists(results_file)):
        with open(results_file) as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    params_file = os.path.join(out_dir, dataset_name + "_" + net + "_params.json")
    if (os.path.exists(params_file)):
        with open(params_file) as f:
            params_dict = json.load(f)
    else:
        params_dict = {}

    if net == "GCN":
        if use_grid_search:
            param_combinations = utils.generate_combinations(parameters_grid_GCN)
        else:
            param_combinations = [parameters_GCN]
    elif net == "GAT":
        if use_grid_search:
            param_combinations = utils.generate_combinations(parameters_grid_GAT)
        else:
            param_combinations = [parameters_GAT]
    else:
        if use_grid_search:
            param_combinations = utils.generate_combinations(parameters_grid_SAGE)
        else:
            param_combinations = [parameters_SAGE]

    i = 1
    for params in param_combinations:

        logdir = os.path.join("logs", "{}-{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(params.items())))
        ))

        writer = SummaryWriter(log_dir=logdir)

        print("\n " + net + ", (iteration " + str(i) + " over " + str(
            len(param_combinations)) + ") - Testing parameters: ")
        i += 1
        for key, value in params.items():
            print(f"{key}: {value}", end="\n")
        print("--------------------------------\n")

        if net == "SAGE":
            batch_generation = True
            num_batch_neighbors = params["num_batch_neighbors"]
            batch_size = params["batch_size"]
        else:
            batch_generation = False
            num_batch_neighbors = []
            batch_size = None

        # ************************************ CLASSIFICATION 1 ************************************

        input_size = classification_dataset.num_features
        hidden_channels = params["hidden_channels"]
        output_size = classification_dataset.num_classes # params["embedding_size"]
        dropout = params["dropout"]

        if net == "GCN":
            network = model.GCN(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels,
                                dropout=dropout)
        elif net == "GAT":
            heads = params["heads"]
            heads_out = params["heads_out"]
            network = model.GAT(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels,
                                heads=heads, heads_out=heads_out, dropout=dropout)
        else:
            network = model.Graph_SAGE(input_size=input_size, embedding_size=output_size,
                                       hidden_channels=hidden_channels, dropout=dropout)


        # print(summary(network, classification_dataset.data.x, classification_dataset.data.edge_index, max_depth=5))


        # ************************************ LINK PREDICTION ************************************

        print("\n************************* TRAINING LINK PREDICTION *************************")

        network = network.to(device)

        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

        # run the training
        epochs_linkpred = params["epochs_linkpred"]

        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_linkpred, eta_min=lr / 1e3)

        epochs = epochs_linkpred
        writer_info = {'dataset_name': 'no_mlp'+dataset_name, 'training_step': 'link_pred', 'model_name': net,
                       'second_tr_e': None, 'starting_epoch': 0}
        results_linkpred = engine.train_link_prediction(network, train_ds, val_ds, criterion, optimizer, epochs, writer,
                                     writer_info,
                                     device, batch_generation, num_batch_neighbors, batch_size, lr_schedule)

        # writer_info = {'dataset_name': 'no_mlp'+dataset_name, 'training_step': 'link_pred', 'model_name': net,
        #                'second_tr_e': epochs, 'starting_epoch': 0 + epochs}
        # optimizer = torch.optim.Adam(network.parameters(), lr=lr_schedule.get_lr()[0], weight_decay=weight_decay)
        # epochs = epochs_linkpred - epochs
        # engine.train_link_prediction(network, train_ds, val_ds, criterion, optimizer, epochs, writer,
        #                              writer_info,
        #                              device, batch_generation, num_batch_neighbors, batch_size, lr_schedule)

        print() 
        print("*****************************************************************************\n")

        # ************************************ CLASSIFICATION 2 ************************************

        print("************************** TRAINING CLASSIFICATION **************************")

        model_classification2 = network.to(device)

        criterion = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.1)

        # run the training
        epochs_classification2 = params["epochs_classification2"]
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_classification2, eta_min=lr / 1e2)

        writer_info = {'dataset_name': 'no_mlp'+dataset_name, 'training_step': 'class2', 'model_name': net,
                           'second_tr_e': epochs, 'starting_epoch': epochs_linkpred}
        optimizer = torch.optim.Adam(model_classification2.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        results_class2b = engine.train_classification(model_classification2, classification_dataset.data,
                                                      classification_dataset.data, criterion,
                                                      optimizer, epochs_classification2, writer, writer_info, device,
                                                      batch_generation,
                                                      num_batch_neighbors, batch_size, lr_schedule)
        # print()
        # print("\nCLASSIFICATION 2b RESULTS")
        # for k, v in results_class2b.items():
        #     print(k + ":" + str(v[-1]))
        # print("****************************************************** \n")
        # _, acc2 = engine.eval_classifier(model_classification2, criterion, classification_dataset.data, False,
        #                                  batch_generation, device, num_batch_neighbors, batch_size)
        # print("test acc with LinkPrediction:", acc2)

        print()
        print("*****************************************************************************")

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
            json.dump(params_dict, f, indent=4)

        """
        # Save results of the training
        results_class1_list = []
        for k, r in results_class1.items():
            results_class1_list.append((k, r))
        """

        # results_class2b_list = []
        # for k, r in results_class2b.items():
        #     results_class2b_list.append((k, r))

        # results_dict[key] = [("results_class2b", results_class2b_list)]

        # params["hidden_sizes_mlp_class1"] = str(params["hidden_sizes_mlp_class1"])
        # params["hidden_sizes_mlp_link_pred"] = str(params["hidden_sizes_mlp_link_pred"])
        # params["hidden_sizes_mlp_class2"] = str(params["hidden_sizes_mlp_class2"])
        # if(net == "SAGE"):
        #     params["num_batch_neighbors"] = str(params["num_batch_neighbors"])

        # for k, r in results_class2b.items():
        #     results_class2b[k] = str(r)

        # writer.add_hparams(params, results_class2b)

        test_loss, test_acc = engine.eval_classifier(model_classification2, criterion, classification_dataset.data,False,batch_generation,device,num_batch_neighbors,batch_size)
        results_class2b["test_loss"] = [test_loss]
        results_class2b["test_acc"] = [test_acc]

        if key in results_dict.keys():
            for k, r in results_class2b.items():
                results_dict[key][k].append(r[-1])
        else:
            results_dict[key] = {}
            for k, r in results_class2b.items():
                results_dict[key][k] = [r[-1]]

        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=4)
        
        print("\nLink prediction val accuracy: ", results_linkpred["val_acc"][-1])
        print("Classification 2b val accuracy: ", results_class2b["val_acc"][-1])
        # print("\nTest accuracy: ", test_acc)

        # _, test_acc = engine.eval_classifier(model_classification2, criterion, classification_dataset.data,False,batch_generation,device,num_batch_neighbors,batch_size)
        # print("\nTest accuracy: ", test_acc)
        
        print()
        print("*****************************************************************************")


    # if use_grid_search:
    #     num_best_runs = 20
    #     filename = dataset_name + "_" + net + "_best_runs.txt"
    #     filepath = os.path.join(out_dir, filename)
    #     sorted_accuracies = get_best_params.find_best_params(dataset_name, net, results_dict, params_dict,
    #                                                          num_best_runs, print_output=False, save_output=True,
    #                                                          file_name=filepath)

    #     filename = dataset_name + "_" + net + "_params_counter.txt"
    #     filepath = os.path.join(out_dir, filename)
    #     get_best_params.count_params_in_best_runs(sorted_accuracies, num_best_runs, filepath)