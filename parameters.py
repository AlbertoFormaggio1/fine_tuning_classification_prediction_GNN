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
parameters_grid_GAT = {
    "embedding_size": [64, 128, 256],
    "hidden_channels": [16, 32], 
    "heads": [1, 8],
    "dropout": [0.4, 0.6, 0.8], #trovati in tabella pag 19 per pubmed
    "hidden_sizes_mlp_class1": [[20]],
    "hidden_sizes_mlp_link_pred": [[5]],
    "hidden_sizes_mlp_class2": [[5]],
    "dropout_mlp_class1": [0],
    "dropout_mlp_link_pred": [0],
    "dropout_mlp_class2": [0],
    "link_pred_out_size_mlp" : [16],
    "epochs_classification1": [50, 100],
    "epochs_linkpred": [25, 50],
    "net_freezed_linkpred": [0.4, 0.6],
    "epochs_classification2": [25, 50],
    "net_freezed_classification2": [0.4, 0.6],
}

parameters_GAT = {
    "embedding_size": 64,
    "hidden_channels": 16,
    "heads": 8, # in più rispetto a GCN e SAGE
    "dropout": 0.6,
    "hidden_sizes_mlp_class1": [20],
    "hidden_sizes_mlp_link_pred": [5],
    "hidden_sizes_mlp_class2": [5],
    "dropout_mlp_class1": 0,
    "dropout_mlp_link_pred": 0,
    "dropout_mlp_class2": 0,
    "link_pred_out_size_mlp" : 16,
    "epochs_classification1": 100,
    "epochs_linkpred": 50,
    "net_freezed_linkpred": 0.6,
    "epochs_classification2": 50,
    "net_freezed_classification2": 0.6,
}

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


# epochs_classification1 = [50, 100]
# epochs_linkpred = [25, 50]
# net_freezed_linkpred = [0.4, 0.6]
# epochs_classification2 = [25, 50]
# net_freezed_classification2 = [0.4, 0.6]

# aggiungere anche lr e decay rate e hidden_sizes?
# diversificare i parametri di rete/mlp tra class1/linkpred/class2 ?