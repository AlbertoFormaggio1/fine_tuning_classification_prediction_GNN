#GCN
#max 200 epochs, adam, lr = 0.01, earling stopping = 10, pag 6
parameters_grid_GCN = {
    "embedding_size": [32], #[32, 48, 64],
    "hidden_channels": [32], #[16, 32], 
    "dropout": [0.6, 0.7], #[0.3, 0.6], # 0 pag 6
    "hidden_sizes_mlp_class1": [[10]], #[[10], [15]],
    "hidden_sizes_mlp_link_pred": [[15]], #[[10], [15]],
    "hidden_sizes_mlp_class2": [[10]], #[[10], [15]],
    "dropout_mlp_class1": [0.2],
    "dropout_mlp_link_pred": [0.2],
    "dropout_mlp_class2": [0.2],

    "epochs_classification1": [50, 75],
    "epochs_linkpred": [25, 50],
    "net_freezed_linkpred": [0.4, 0.6],
    "epochs_classification2": [50, 75],
    "net_freezed_classification2": [0.4, 0.6]
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

    "epochs_classification1": 75,
    "epochs_linkpred": 50,
    "net_freezed_linkpred": 0.6,
    "epochs_classification2": 100,
    "net_freezed_classification2": 0.4
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

    "epochs_classification1": [50, 100],
    "epochs_linkpred": [25, 50],
    "net_freezed_linkpred": [0.4, 0.6],
    "epochs_classification2": [25, 50],
    "net_freezed_classification2": [0.4, 0.6]
}

parameters_GAT = {
    "embedding_size": 64,
    "hidden_channels": 16,
    "heads": 8,
    "dropout": 0.6,
    "hidden_sizes_mlp_class1": [20],
    "hidden_sizes_mlp_link_pred": [5],
    "hidden_sizes_mlp_class2": [5],
    "dropout_mlp_class1": 0,
    "dropout_mlp_link_pred": 0,
    "dropout_mlp_class2": 0,

    "epochs_classification1": 100,
    "epochs_linkpred": 50,
    "net_freezed_linkpred": 0.6,
    "epochs_classification2": 50,
    "net_freezed_classification2": 0.6    
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
    "num_batch_neighbors": [[10, 4], [15, 6]],

    "epochs_classification1": [50, 100],
    "epochs_linkpred": [25, 50],
    "net_freezed_linkpred": [0.4, 0.6],
    "epochs_classification2": [25, 50],
    "net_freezed_classification2": [0.4, 0.6]
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

    "epochs_classification1": 100,
    "epochs_linkpred": 50,
    "net_freezed_linkpred": 0.6,
    "epochs_classification2": 50,
    "net_freezed_classification2": 0.6
}

lr=0.01
weight_decay=5e-4

# epochs_classification1 = [50, 100]
# epochs_linkpred = [25, 50]
# net_freezed_linkpred = [0.4, 0.6]
# epochs_classification2 = [25, 50]
# net_freezed_classification2 = [0.4, 0.6]

# aggiungere anche lr e decay rate e hidden_sizes?
# diversificare i parametri di rete/mlp tra class1/linkpred/class2 ?