#GCN
#max 200 epochs, adam, lr = 0.01, earling stopping = 10, pag 6
lr = 0.1
weight_decay = 0.001
parameters_grid_GCN = {
    "x": [1,2,3,4,5],
    "embedding_size": [32], #[32, 48, 64],
    "hidden_channels": [16], #[16, 32],
    "dropout": [0.5], # [0.6, 0.7], #[0.3, 0.6], # 0 pag 6
    "hidden_sizes_mlp_class1": [[10]], #[[10], [15]],
    "hidden_sizes_mlp_link_pred": [[10]], #[[10], [15]],
    "hidden_sizes_mlp_class2": [[10]], #[[10], [15]],
    "dropout_mlp_class1": [0],
    "dropout_mlp_link_pred": [0],
    "dropout_mlp_class2": [0],
    "link_pred_out_size_mlp" : [16],
    "epochs_classification1": [50],
    "epochs_linkpred": [150],
    "net_freezed_linkpred": [0.5],
    "epochs_classification2": [100],
    "net_freezed_classification2": [0.5] #, 0.6],
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

# All parameters with the "with MLP" tag are used only when running a 
# configuration with an MLP layer at the end of the network
parameters_grid_GAT = {
    "number_of_runs": [1, 2, 3, 4, 5],
    "embedding_size": [8],              # with MLP
    "hidden_channels": [8],
    "heads": [8],
    "heads_out": [1],
    "dropout": [0.6],
    "hidden_sizes_mlp_class1": [[16]],      # with MLP
    "hidden_sizes_mlp_link_pred": [[16]],   # with MLP
    "hidden_sizes_mlp_class2": [[16]],      # with MLP
    "dropout_mlp_class1": [0.4],        # with MLP
    "dropout_mlp_link_pred": [0.4],     # with MLP
    "dropout_mlp_class2": [0.4],        # with MLP
    "link_pred_out_size_mlp" : [16],    # with MLP
    "epochs_classification1": [200],    # with MLP
    "epochs_linkpred": [200],
    "net_freezed_linkpred": [0.0],      # with MLP
    "epochs_classification2": [200],
    "net_freezed_classification2": [0.4]    # with MLP
}

parameters_GAT = {
    "embedding_size": 8,                # with MLP
    "hidden_channels": 8,
    "heads": 8,
    "heads_out": 1,
    "dropout": 0.6,
    "hidden_sizes_mlp_class1": [16],        # with MLP
    "hidden_sizes_mlp_link_pred": [16],     # with MLP
    "hidden_sizes_mlp_class2": [16],        # with MLP
    "dropout_mlp_class1": 0.4,          # with MLP
    "dropout_mlp_link_pred": 0.4,       # with MLP
    "dropout_mlp_class2": 0.4,          # with MLP
    "link_pred_out_size_mlp" : 16,      # with MLP
    "epochs_classification1": 200,      # with MLP
    "epochs_linkpred": 200,
    "net_freezed_linkpred": 0.0,        # with MLP
    "epochs_classification2": 200,
    "net_freezed_classification2": 0.4      # with MLP
}
lr = 0.01
weight_decay = 0.001


# SAGE
parameters_grid_SAGE = {
    "hidden_channels": [32],
    "dropout": [0.7],
    "num_batch_neighbors": [[5,2]],
    "epochs_linkpred": [8],
    "epochs_classification2": [200],
    "batch_size": [32],
}

parameters_SAGE = {
    "embedding_size": 32,
    "hidden_channels": 64,
    "dropout": 0.5,
    "hidden_sizes_mlp_class1": [16],
    "hidden_sizes_mlp_link_pred": [16],
    "hidden_sizes_mlp_class2": [16],
    "dropout_mlp_class1": 0.4,
    "dropout_mlp_link_pred": 0.4,
    "dropout_mlp_class2": 0.4,
    "num_batch_neighbors": [5, 10],
    "link_pred_out_size_mlp" : 16,
    "epochs_classification1": 50,
    "epochs_linkpred": 50,
    "net_freezed_linkpred": 0.6,
    "epochs_classification2": 50,
    "net_freezed_classification2": 0.6,
    "batch_size": 8
}

parameters_grid_SAGE_pubmed = {
    "embedding_size": [32], #32
    "hidden_channels": [256], #512
    "dropout": [0.7], #0.2
    "hidden_sizes_mlp_class1": [[32]],
    "hidden_sizes_mlp_link_pred": [[32]],
    "hidden_sizes_mlp_class2": [[32]],
    "dropout_mlp_class1": [0.5],
    "dropout_mlp_link_pred": [0.5],
    "dropout_mlp_class2": [0.1],
    "num_batch_neighbors": [[5,2]],
    "link_pred_out_size_mlp": [256],
    "epochs_classification1": [50],
    "epochs_linkpred": [100],
    "net_freezed_linkpred": [0.4],
    "epochs_classification2": [90],
    "net_freezed_classification2": [0.4],
    "batch_size": [1024], # Done
}

parameters_grid_SAGE_cora = {
    "embedding_size": [16], #32
    "hidden_channels": [16], #512
    "dropout": [0.5], #0.2
    "hidden_sizes_mlp_class1": [[32]],
    "hidden_sizes_mlp_link_pred": [[32]],
    "hidden_sizes_mlp_class2": [[32]],
    "dropout_mlp_class1": [0],
    "dropout_mlp_link_pred": [0.2],
    "dropout_mlp_class2": [0.1],
    "num_batch_neighbors": [[5,2]],
    "link_pred_out_size_mlp": [16],
    "epochs_classification1": [50],
    "epochs_linkpred": [50],
    "net_freezed_linkpred": [0.4],
    "epochs_classification2": [85],
    "net_freezed_classification2": [0.45],
    "batch_size": [1024], # Done
}

parameters_grid_SAGE_citeseer = {
    "embedding_size": [32], #32
    "hidden_channels": [32], #512
    "dropout": [0.7], #0.2
    "hidden_sizes_mlp_class1": [[32]],
    "hidden_sizes_mlp_link_pred": [[32]],
    "hidden_sizes_mlp_class2": [[32]],
    "dropout_mlp_class1": [0.1],
    "dropout_mlp_link_pred": [0.2],
    "dropout_mlp_class2": [0],
    "num_batch_neighbors": [[5,2]],
    "link_pred_out_size_mlp": [256],
    "epochs_classification1": [50],
    "epochs_linkpred": [100],
    "net_freezed_linkpred": [0.4],
    "epochs_classification2": [150],
    "net_freezed_classification2": [0.15],
    "batch_size": [1024], # Done
}