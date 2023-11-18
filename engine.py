import torch
from torch_geometric.datasets import Planetoid
from tqdm.auto import tqdm


def train(model, train_ds, val_ds, loss_fn: torch.nn.Module,
          opt: torch.optim.Optimizer, epochs: int):
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, ds=train_ds, loss_fn=loss_fn, opt=opt)
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
