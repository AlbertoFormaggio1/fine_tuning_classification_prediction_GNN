## GNN Integration for Node Classification and Link Prediction

### Research Overview:

This repository investigates the collaborative dynamics between node classification and link prediction in the realm of Graph Neural Networks (GNNs). The primary goal is to evaluate the potential advantages gained by integrating these tasks within a graph, as opposed to solely utilizing GNNs for node classification. By merging the predictive capabilities of link prediction with the node classification features of GNNs, the aim is to enhance the overall performance and effectiveness of graph-based models.

### Implementation Details:

- In the "main.py" file, the combination of GNN + MLP (Multi-Layer Perceptron) + Link Prediction is implemented.
- For an alternative approach without MLP, refer to the "main_no_MLP.py" file, utilizing the combination of GNN + Link Prediction.

### Configuration:

Before running the code, navigate to the ***COMMANDS*** section (lines 25-30) in either "main.py" or "main_no_mlp.py" to set the preferred GNN (GCN, GAT, SAGE) and dataset (cora, citeseer, pubmed). For hyperparameter testing, enable the "use_grid_search" option. Adjust hyperparameters in the "parameters.py" file, filling the "parameters_[GNN]" dictionary for specific sets, or use "parameters_grid_[GNN]" for comprehensive hyperparameter combination testing by setting "use_grid_search = True".