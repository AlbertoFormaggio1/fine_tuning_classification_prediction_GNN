import torch
from tqdm.auto import tqdm
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

def train(model, train_ds, val_ds, loss_fn: torch.nn.Module,
          opt: torch.optim.Optimizer, epochs: int, batch_generation: bool = False):
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }

    # whether we need to use some loader to generate batches or we use the full-batch approach
    if batch_generation:
        # We keep the 25 neighbors of each node and then 10 neighbors for each of them
        # They trained on 10 epochs for the fully supervised sampling
        train_batches = NeighborLoader(train_ds, [25, 10], batch_size=512, input_nodes=train_ds.train_mask, shuffle=True)
    else:
        train_batches = [train_ds]

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = .0, .0
        batch_num = 0
        for batch in train_batches:
            cur_train_loss, cur_train_acc = train_step(model=model, ds=batch, loss_fn=loss_fn, opt=opt)
            if batch_generation:
                batch_size = batch.batch_size
            else:
                batch_size = torch.sum(batch.train_mask).item()
            batch_num += batch_size
            train_loss += cur_train_loss * float(batch_size)
            train_acc += cur_train_acc * float(batch_size)

        train_loss /= batch_num
        train_acc /= batch_num

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)

        # every 10 epochs see the improvement on the validation set
        if epoch % 10 == 0:
            val_loss, val_acc = eval(model, loss_fn, val_ds, val_ds.val_mask)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)

    return results

def eval(model, loss_fn, ds, mask):
    with torch.no_grad():
        out = model(ds.x, ds.edge_index)
        loss = loss_fn(out[mask], ds.y[mask]).item()
        cls = out.argmax(dim=-1)
        acc = torch.sum(cls[mask] == ds.y[mask]) / torch.sum(mask)

    return loss, acc.item()


def train_step(model: torch.nn.Module, ds, loss_fn: torch.nn.Module,
               opt: torch.optim.Optimizer):
    model.train()  # Set the model in training mode
    opt.zero_grad() # Reset the gradient
    out = model(ds.x, ds.edge_index)  # Compute the response of the model
    loss = loss_fn(out[ds.train_mask], ds.y[ds.train_mask])  # Compute the loss based on training nodes
    loss.backward()  # Propagate the gradient
    opt.step()  # Update the weights

    train_loss = loss.item()  # Get the loss
    # Compute the classification accuracy
    train_cls = out.argmax(dim=-1)
    train_acc = torch.sum(train_cls[ds.train_mask] == ds.y[ds.train_mask]) / torch.sum(ds.train_mask)

    return train_loss, train_acc.item()


def train_link_prediction(model, train_ds, loss_fn: torch.nn.Module,
          opt: torch.optim.Optimizer, epochs: int):

    for epoch in tqdm(range(epochs)):
        model.train()
        opt.zero_grad()
        # Computing first the embeddings with message passing on the edges that are already existing
        # in the graph
        z = model(train_ds.x, train_ds.edge_index)

        # For every epoch perform a round of negative sampling.
        # This array will return edges not already present in edge_index.
        # The number of nodes is given by num_nodes
        # The number of negative edges to generate is the same as the number of edges in the original graph, this way the predictor is unbiased
        neg_edge_index = negative_sampling(
            edge_index=train_ds.edge_index, num_nodes=train_ds.num_nodes,
            num_neg_samples=train_ds.edge_label_index.size(1), method='sparse')

        # The edge_label for the edges that are already in the graph will be 1
        # The edge_label for the edges we just created instead will be 0

        # concatenating on the last dimensions since we're adding more edges
        edge_label_index = torch.cat([train_ds.edge_label_index, neg_edge_index], dim=1)
        # Concatenating along the 1st (and only dimension) the label of the negative edges (thus, 0)
        edge_label = torch.cat([train_ds.edge_label,
                                train_ds.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

        #out = model.decode(z, edge_label_index).view(-1)
        # Let's understand first what decode returns
        out = model.decode(z, edge_label_index)
        loss = loss_fn(out, edge_label)
        loss.backward()
        opt.step()

    return loss.item()

@torch.no_grad()
def eval_predictor(model, data):
    model.eval()
    z = model(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())