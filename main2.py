import torch
import os
import json
import load_dataset
import engine
import model
import torch_geometric.transforms as T

# select the device on which you should run the computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ************************************ PARAMETERS ************************************

net_name = "GAT"
dataset_name = "cora"

results_file = os.path.join(dataset_name + "_results.json")
import json
if(os.path.exists(results_file)):
    with open(results_file) as f:
        final_dict = json.load(f)
else:
    final_dict = {}

#GCN
# parameters = [("embedding_size", 5), 
#     ("hidden_channels", 16), 
#     ("heads", 8),
#     ("dropout", 0.6)]

#GAT
parameters = [("embedding_size", 5), 
    ("hidden_channels", 16), 
    ("dropout", 0.5)]

# SAGE
# parameters = [("embedding_size", 5), 
#     ("hidden_channels", 512), 
#     ("dropout", 0.5)]

epochs = 10

# aggiungere anche lr e decay rate?

# ************************************ DATASET ************************************

# Normalize the features and put it on the appropriate device
transform_classification = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device)
])

datasets = {}

# Load the 3 datasets and apply the transform needed
datasets['cora'] = load_dataset.load_ds('Cora', transform_classification)
datasets['citeseer'] = load_dataset.load_ds('CiteSeer', transform_classification)
datasets['pubmed'] = load_dataset.load_ds('PubMed', transform_classification)

# print the information for each dataset
for ds in datasets.values():
    load_dataset.print_ds_info(ds)
    print('\n#################################\n')

# Again, for debug purposes i extracted only one dataset. This should be transformed in a for loop done over all the
# possible values of the dictionary (in this case, the datasets)
dataset = datasets[dataset_name]

# Create a model, also this can be transformed in a for loop that tries all possible architectures
model = model.GAT(dataset.num_features, dataset.num_classes)

# define the loss function and the optimizer. The learning rate is found on papers, same goes for the learning rate decay
# and the weight decay
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion => CrossEntropyLoss in the case of classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# run the training
results = engine.train(model, dataset.data, dataset.data, criterion, optimizer, 10, False)

# print the results of the training (training loss, validation loss, training accuracy, validation accuracy)
for k, r in results.items():
    params = ""
    for p in parameters:
        params = params + "_" + str(p[1])
    key = net_name + "_" + k + "_" + params
    final_dict[key] = r

with open(results_file, "w") as f:
    json.dump(final_dict, f, indent = 4)





#LINK PREDICTION

# # Create the transform to apply the dataset
# # In this case we also perform a random link split since we will use only a subset of the edges for the training.
# # the remaining edges will be reserved to test and validation (10% and 5%, respectively).
# # We don't add negative train samples since we will handle that during the training stage
# transform_prediction = T.Compose([
#     T.NormalizeFeatures(),
#     T.ToDevice(device),
#     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                         add_negative_train_samples=False)
# ])

# datasets = {}

# # Load the datasets as before
# datasets['cora'] = load_dataset.load_ds('Cora', transform_prediction)
# datasets['citeseer'] = load_dataset.load_ds('CiteSeer', transform_prediction)
# datasets['pubmed'] = load_dataset.load_ds('PubMed', transform_prediction)

# dataset = datasets['cora']
# # Get the 3 splits
# train_ds, val_ds, test_ds = dataset[0]

# model = model.GCN(dataset.num_features, dataset.num_classes)

# # Define loss criterion => Binary Cross Entropy for link prediction
# # We need to predict 1 if the link is present, 0 otherwise
# criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# engine.train_link_prediction(model, train_ds, criterion, optimizer, 200)

# acc = engine.eval_predictor(model, val_ds)
# print(acc)





"""
# Crea modello MLP 1
gat = model.GAT()
mlp_1 = model.MLP()
#Gat e mlp randomly initialized
model_1 = GAT_MLP(gat, mlp)

# Crea loss e optimizer che ottimizzi tutti i parametri del modello
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim(model_1.parameters())
# Allenamento classificazione (train_classifier)
engine.train(model_1,...)

################## FINE ALLENAMENTO 1 CLASSIFICAZIONE

mlp_2 = model.MLP()
# Riciclando gat precendente => Non vuoi che i pesi inizializzati random dell'MLP vadano a cambiare erroneamente
# I pesi della GAT
# mlp_2 Randomly init
model_2 = GAT_MLP(gat, mlp_2)

loss = torch.nn.BCEWithLogitsLoss()
# Per le prime epoche_totali / 2 (numero a caso) ==> allenare e settare i pesi della MLP
optimizer = torch.optim(mlp_2.parameters(), ...)
engine.train_link_prediction(model_2, optimizer, epochs=EPOCHE TOTALI / 2) 

# Una volta che i pesi della MLP sono stati settati correttamente, fai il fine-tuning della rete intera
optimizer = torch.optim(model_2.parameters(),...)
engine.train_link_prediction(model_2, optimizer, epochs=RESTANTI EPOCHE PER RAGGIUNGERE IL TOTALE)

################### FINE ALLENAMENTO 2

# Si riutilizza model_1 (siccome GAT ha i pesi aggiornati) per fare la classificazione
# Fare comunque qualche epoca di fine tuning per avere poi i risultati (fare Copia incolla)

# Oppure creare una nuova MLP e fare come si era fatto con la seconda iterazione della link prediction
# prima fine tuning solo MLP e poi allenare MLP + GAT

"""

# Embedding size
# hidden size (MLP)
# Lr (Dai paper)
# (Tutti parametri messi nei costruttori)
# Epoche