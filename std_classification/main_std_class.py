import datetime
import json
import os
import re

import torch
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard

import engine_std_class as engine
import load_dataset_std_class as load_dataset
import model_std_class as model
import utils_std_class as utils
import parameters_std_class as parameters

random_seed = 42
#torch.manual_seed(random_seed)
#torch.cuda.manual_seed_all(random_seed)

# select the device on which you should run the computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#************************************** COMMANDS ************************************

use_grid_search = False #False
dataset_name = "cora"  # cora - citeseer - pubmed
nets = ["GAT"]  # GCN - GAT - SAGE

# ************************************ PARAMETERS ************************************

#GCN
parameters_grid_GCN = parameters.parameters_grid_GCN
parameters_GCN = parameters.parameters_GCN

#GAT
# parameters_grid_GAT = parameters.parameters_grid_GAT
# parameters_GAT = parameters.parameters_GAT

# SAGE
#parameters_grid_SAGE = parameters.parameters_grid_SAGE
#parameters_SAGE = parameters.parameters_SAGE

# Others
# lr = parameters.lr
# weight_decay = parameters.weight_decay


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

# ************************************ TRAINING ************************************

for net in nets:

    out_dir = "std_class_results"# + dataset_name + "_" + net
    os.makedirs(out_dir, exist_ok=True)

    results_file = os.path.join(out_dir, dataset_name + "_" + net + "_results.json")
    if(os.path.exists(results_file)):
        with open(results_file) as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    # params_file = os.path.join(out_dir, dataset_name + "_" + net + "_params.json")
    # if(os.path.exists(params_file)):
    #     with open(params_file) as f:
    #         params_dict = json.load(f)
    # else:
    #     params_dict = {}

    if net == "GCN":
        if use_grid_search:
            param_combinations = utils.generate_combinations(parameters_grid_GCN)
        else:
            param_combinations = [parameters_GCN]
    elif net == "GAT":
        if dataset_name == "cora":
            param_combinations = [parameters.parameters_GAT_cora]
        elif dataset_name == "citeseer":
            param_combinations = [parameters.parameters_GAT_citeseer]
        elif dataset_name == "pubmed":
            param_combinations = [parameters.parameters_GAT_pubmed]
    else:
        # For sage when running this results are different than the ones reported in the paper
        # since I removed label smoothing and cosineAnnealing
        if dataset_name == "cora":
            param_combinations = [parameters.parameters_SAGE_cora]
        elif dataset_name == "citeseer":
            param_combinations = [parameters.parameters_SAGE_citeseer]
        elif dataset_name == "pubmed":
            param_combinations = [parameters.parameters_SAGE_pubmed]

    i = 1
    for params in param_combinations:

        logdir = os.path.join("logs", "{}-{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(params.items())))
        ))
        
        writer = SummaryWriter(log_dir=logdir)

        print("\n " + net + ", (iteration " + str(i) + " over " + str(len(param_combinations)) + ") - Testing parameters: ")
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

        # ************************************ CLASSIFICATION ************************************

        input_size = classification_dataset.num_features
        hidden_channels = params["hidden_channels"]
        output_size = classification_dataset.num_classes #params["embedding_size"]
        dropout = params["dropout"]
        lr = params["lr"]
        weight_decay = params["weight_decay"]

        if net == "GCN":
            network = model.GCN(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels, dropout=dropout)
        elif net == "GAT":
            heads = params["heads"]
            heads_out = params["heads_out"]
            network = model.GAT(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels, heads=heads, heads_out=heads_out, dropout=dropout)
        else:
            network = model.Graph_SAGE(input_size=input_size, embedding_size=output_size, hidden_channels=hidden_channels, dropout=dropout)

        network = network.to(device)

        # define the loss function and the optimizer. The learning rate is found on papers, same goes for the learning rate decay
        # and the weight decay
        criterion = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.1)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

        writer_info = {'dataset_name': "std_class" + dataset_name, 'training_step': 'class', 'model_name': net}

        # run the training
        epochs = params["epochs"]
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr / 1e3)
        results_class = engine.train_classification(network, classification_dataset.data, classification_dataset.data, criterion,
                                                     optimizer, epochs, writer, writer_info, device, batch_generation,
                                                     num_batch_neighbors, batch_size, lr_schedule)

        print()
        print("*************** CLASSIFICATION RESULTS ***************")

        train_loss = results_class["train_loss"][-1]
        train_acc = results_class["train_acc"][-1]
        print("\n - TRAIN loss: " + str(train_loss))
        print(" - TRAIN acc : " + str(train_acc))

        val_loss = results_class["val_loss"][-1]
        val_acc = results_class["val_acc"][-1]
        print("\n - VALIDATION loss: " + str(val_loss))
        print(" - VALIDATION acc : " + str(val_acc))

        test_loss, test_acc = engine.eval_classifier(network, criterion, classification_dataset.data,False,batch_generation,device,num_batch_neighbors,batch_size)
        print("\n - TEST loss: " + str(test_loss))
        print(" - TEST acc : " + str(test_acc))

        results_class["test_loss"] = [test_loss]
        results_class["test_acc"] = [test_acc]

        print("*****************************************************")

        # Set key to use in dictionaries
        key = net + "||"
        for k, v in params.items():
            key = key + k[0:3] + "_" + str(v) + "/"
        
        if key in results_dict.keys():
            for k, r in results_class.items():
                results_dict[key][k].append(r[-1])
        else:
            results_dict[key] = {}
            for k, r in results_class.items():
                results_dict[key][k] = [r[-1]]
        
        test_acc_tensor = torch.tensor(results_dict[key]["test_acc"])
        results_dict[key]["mean"] = torch.mean(test_acc_tensor).tolist() if len(results_dict[key]["test_acc"]) > 1 else results_dict[key]["test_acc"][0]
        results_dict[key]["var"] = torch.std(test_acc_tensor).tolist() if len(results_dict[key]["test_acc"]) > 1 else 0
        
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent = 4)

