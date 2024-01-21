#GCN
#max 200 epochs, adam, lr = 0.01, earling stopping = 10, pag 6
parameters_grid_GCN = {
    "embedding_size": [32], #[32, 48, 64],
    "hidden_channels": [32], #[16, 32], 
    "dropout": [0.2], # [0.6, 0.7], #[0.3, 0.6], # 0 pag 6
    "hidden_sizes_mlp_class1": [[10]], #[[10], [15]],
    "hidden_sizes_mlp_link_pred": [[15]], #[[10], [15]],
    "hidden_sizes_mlp_class2": [[10]], #[[10], [15]],
    "dropout_mlp_class1": [0.2],
    "dropout_mlp_link_pred": [0.2],
    "dropout_mlp_class2": [0.2],
    "link_pred_out_size_mlp" : [16],
    "epochs_classification1": [50, 75],
    "epochs_linkpred": [25, 50],
    "net_freezed_linkpred": [0.4, 0.6],
    "epochs_classification2": [50, 75],
    "net_freezed_classification2": [0.4] #, 0.6],
}

parameters_GCN = {
    "embedding_size": 64,
    "hidden_channels": 16,
    "dropout": 0.4,
    "hidden_sizes_mlp_class1": [15],
    "hidden_sizes_mlp_link_pred": [10],
    "hidden_sizes_mlp_class2": [15],
    "dropout_mlp_class1": 0.3,
    "dropout_mlp_link_pred": 0,
    "dropout_mlp_class2": 0.3,
    "link_pred_out_size_mlp" : 16,
    "epochs_classification1": 75,
    "epochs_linkpred": 50,
    "net_freezed_linkpred": 0.6,
    "epochs_classification2": 100,
    "net_freezed_classification2": 0.4,
}

#GAT
parameters_GAT_cora = {
    "hidden_channels": 8,
    "heads": 8,
    "heads_out": 1,
    "dropout": 0.6,     # X
    "epochs": 200,
    "lr" : 0.05,
    "weight_decay" : 0.0005
}

parameters_GAT_citeseer = {
    "hidden_channels": 8,
    "heads": 8,
    "heads_out": 1,
    "dropout": 0.6,     # X
    "epochs": 200,
    "lr" : 0.05,
    "weight_decay" : 0.0005
}

parameters_GAT_pubmed = {
    "hidden_channels": 8,
    "heads": 8,
    "heads_out": 8,
    "dropout": 0.6,     # X
    "epochs": 200,
    "lr" : 0.05,
    "weight_decay" : 0.0001
}

lr = 0.05
weight_decay = 0.0005

# SAGE
parameters_grid_SAGE = {
    "embedding_size": [32],
    "hidden_channels": [64], 
    "dropout": [0.2],
    "hidden_sizes_mlp_class1": [[20]],
    "hidden_sizes_mlp_link_pred": [[5]],
    "hidden_sizes_mlp_class2": [[5]],
    "dropout_mlp_class1": [0],
    "dropout_mlp_link_pred": [0],
    "dropout_mlp_class2": [0],
    "num_batch_neighbors": [[10, 4], [15, 6]], # in più rispetto a GCN e GAT
    "link_pred_out_size_mlp" : [16],
    "epochs_classification1": [50, 100],
    "epochs_linkpred": [25, 50],
    "net_freezed_linkpred": [0.4, 0.6],
    "epochs_classification2": [25, 50],
    "net_freezed_classification2": [0.4, 0.6],
    "batch_size": [32], # in più rispetto a GCN e GAT
}

parameters_SAGE = {
    "embedding_size": 32,
    "hidden_channels": 64,
    "dropout": 0.2,
    "hidden_sizes_mlp_class1": [20],
    "hidden_sizes_mlp_link_pred": [5],
    "hidden_sizes_mlp_class2": [5],
    "dropout_mlp_class1": 0,
    "dropout_mlp_link_pred": 0,
    "dropout_mlp_class2": 0,
    "num_batch_neighbors": [10, 4],
    "link_pred_out_size_mlp" : 16,
    "epochs_classification1": 100,
    "epochs_linkpred": 50,
    "net_freezed_linkpred": 0.6,
    "epochs_classification2": 50,
    "net_freezed_classification2": 0.6,
    "batch_size": 32
}
