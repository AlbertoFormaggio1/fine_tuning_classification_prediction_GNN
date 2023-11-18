import torch
from tqdm.auto import tqdm
from torch_geometric.loader import NeighborLoader

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
                batch_size = torch.sum(batch.train_mask)
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
    model.train()  # Set the model in training phase
    out = model(ds.x, ds.edge_index)  # Compute the response of the model
    loss = loss_fn(out[ds.train_mask], ds.y[ds.train_mask])  # Compute the loss based on training nodes
    loss.backward()  # Propagate the gradient
    opt.step()  # Update the weights

    train_loss = loss.item()  # Get the loss
    # Compute the classification accuracy
    train_cls = out.argmax(dim=-1)
    train_acc = torch.sum(train_cls[ds.train_mask] == ds.y[ds.train_mask]) / torch.sum(ds.train_mask)

    return train_loss, train_acc.item()
