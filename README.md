In this research endeavor, we explore the synergies between node classification and link prediction within the context of graph neural networks (GNNs). Our study aims to assess whether the fusion of these two tasks within a graph can yield advantages over employing GNNs solely for node classification. By integrating the predictive capabilities of link prediction with the node classification capabilities of GNNs, we seek to enhance the overall performance and effectiveness of graph-based models.

In file main.py we use the combination GNN + MLP + Link Prediction,
In file main_no_MLP.py we use the combination GNN + Link Prediction.

Before starting, in section ***COMMANDS*** (row 25-30 of main.py or main_no_mlp.py), set the GNN you want to use (GCN, GAT, SAGE) and the dataset (cora, citeseer, pubmed).
If you want to test a set of hyperparameters, set "use_grid_search = True".
In the file parameters.py insert the hyperparameters that you want to use. 
Fill the dictionary parameters_[GNN] with your set of hyperparameter (and set use_grid_search = False), fill the dictionary parameters_grid_[GNN] instead if you want to test all the combinations of hyperparameters (and set use_grid_search = False)
