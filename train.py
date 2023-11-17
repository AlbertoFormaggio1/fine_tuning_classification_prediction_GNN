import torch
from torch_geometric.datasets import Planetoid
from tqdm.auto import tqdm


def train(model, train_ds , test_ds, loss_fn : torch.nn.Module ,
          opt: torch.optim.Optimizer, epochs: int):
    
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, ds=train_ds, loss_fn=loss_fn, opt=opt)
        print(train_loss, train_acc)


def eval(model):
    pass

def train_step(model: torch.nn.Module, ds, loss_fn: torch.nn.Module,
               opt: torch.optim.Optimizer):

    model.train()   #Set the model in training phase
    train_loss, train_acc = 0, 0    # Initialize loss and accuracy
    out = model(ds.x)
    loss = loss_fn(out[ds.train_mask], ds.y[ds.train_mask]) # Compute the loss based on training nodes
    loss.backward()
    opt.step()

    train_loss = loss.item()
    train_cls = out.argmax(dim=-1)
    train_acc = torch.sum(train_cls[ds.train_mask] == ds.y[ds.train_mask]) / torch.sum(ds.train_mask)


    return train_loss, train_acc