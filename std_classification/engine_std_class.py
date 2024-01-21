from typing import List

import torch
# from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm

# import model


def train_classification(model, train_ds: torch.utils.data.DataLoader, val_ds: torch.utils.data.DataLoader,
                         loss_fn: torch.nn.Module,
                         opt: torch.optim.Optimizer, epochs: int, writer, writer_info, device,
                         batch_generation: bool = False,
                         num_batch_neighbors: List[int] = [25, 10], batch_size: int = None, lr_schedule=None):
    """Trains and tests a PyTorch model.

        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        :param model: A PyTorch model to be trained and tested.
        :param train_ds: A DataLoader instance for the model to be trained on.
        :param val_ds: A DataLoader instance for the model to be tested on.
        :param opt: A PyTorch optimizer to help minimize the loss function.
        :param loss_fn: A PyTorch loss function to calculate loss on both datasets.
        :param epochs: An integer indicating how many epochs to train for.
        :param writer: Tensorboard writer for writing the scalars generated during training
        :param writer_info: information on where to put the scalars generated during training
        :param device: A target device to compute on (e.g. "cuda" or "cpu").
        :param batch_generation: Whether batches with Neighbor Sampling should be generated (usually True for GraphSAGE)
        :param num_batch_neighbors: List containing how many nodes to sample in the NeighborSampling at each hop.
        Example: [25, 10] for each node, it will sample 25 neighbors and for each neighbor sampled, it will sample in turn
        10 neighbors.
        :param batch_size: how many sampled subgraphs for each batch

        Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                      train_acc: [...],
                      test_loss: [...],
                      test_acc: [...]}
    """
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }

    # whether we need to use some loader to generate batches or we use the full-batch approach.
    # GraphSAGE works with generation of batches where you only pick a subset of the original graph
    # The other networks instead don't need that
    if batch_generation:
        # We keep the 25 neighbors of each node and then 10 neighbors for each of them
        # They trained on 10 epochs for the fully supervised sampling
        train_batches = NeighborLoader(train_ds, num_neighbors=num_batch_neighbors, batch_size=batch_size,
                                       input_nodes=train_ds.train_mask,
                                       shuffle=True)
    else:
        train_batches = [train_ds]

    model.train()  # ************************
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = .0, .0
        batch_num = 0
        for batch in train_batches:
            batch = batch.to(device)
            cur_train_loss, cur_train_acc = train_step_classification(model=model, ds=batch, loss_fn=loss_fn, opt=opt)
            if batch_generation:
                # If we generated batches, then we get the size by doing batch.batch_size
                batch_size = batch.x[batch.train_mask].shape[0]
            else:
                # Otherwise, we sum the values to 1 in the training mask (training mask has a 1 for every node to consider in the training set)
                batch_size = torch.sum(batch.train_mask).item()
            batch_num += batch_size
            # Increase the loss and the accuracy propotionally to the size of the batch
            train_loss += cur_train_loss
            train_acc += cur_train_acc

        # Compute the average loss and accuracy
        train_loss /= batch_num
        train_acc /= batch_num

        # Updating the learning rate before doing validation
        if lr_schedule is not None:
            lr_schedule.step()

        # every 10 epochs see the improvement on the validation set
        if epoch % 5 == 0 or epoch == epochs-1:
            val_loss, val_acc = eval_classifier(model, loss_fn, val_ds, True, batch_generation, device,
                                                num_batch_neighbors, batch_size)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)

            for k in results.keys():
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{writer_info["training_step"]}/{k}',
                                  results[k][-1], epoch)

    return results


def train_step_classification(model: torch.nn.Module, ds, loss_fn: torch.nn.Module,
                              opt: torch.optim.Optimizer):
    """
    Performs a training step for the classification task over a single batch.

    :param model: model to be trained
    :param ds: the training set or batch
    :param opt: A PyTorch optimizer to help minimize the loss function.
    :param loss_fn: A PyTorch loss function to calculate loss on both datasets.
    :return: (train_loss, # of correct samples)
    """
    #model.train()  # # ************************
    opt.zero_grad()  # Reset the gradient
    out = model(ds.x, ds.edge_index)  # Compute the response of the model
    loss = loss_fn(out[ds.train_mask], ds.y[ds.train_mask])  # Compute the loss based on training nodes
    loss.backward()  # Propagate the gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()  # Update the weights

    train_loss = loss.item()  # Get the loss
    # Compute the classification accuracy
    train_cls = out.argmax(dim=-1)
    train_acc = torch.sum(train_cls[ds.train_mask] == ds.y[ds.train_mask])

    return train_loss, train_acc.item()


@torch.no_grad()
def eval_classifier(model: torch.nn.Module, loss_fn: torch.nn.Module, ds, is_validation: bool, batch_generation: bool,
                    device: torch.device, num_batch_neighbors: List = [25, 10], batch_size: int = 32):
    """
    Tests the model over the classification task by computing classification accuracy and loss.
    It uses either the nodes in the test or in the validation set depending on the parameter is_validation.

    :param model: model to test
    :param loss_fn: A PyTorch loss function to calculate loss on both datasets.
    :param ds: the full graph dataset
    :param is_validation: True is it should test over the validation set, False if the test is done over the test set
    :param batch_generation: Whether batches with Neighbor Sampling should be generated (usually True for GraphSAGE)
    :param device: the device on which to run inference. 'cuda' or 'cpu'
    :param num_batch_neighbors: List containing how many nodes to sample in the NeighborSampling at each hop.
    Example: [25, 10] for each node, it will sample 25 neighbors and for each neighbor sampled, it will sample in turn
    10 neighbors.
    :param batch_size: how many sampled subgraphs for each batch
    :return: (evaluation_loss, evaluation_accuracy)
    """
    model = model.to(device)
    model.eval()

    if batch_generation:
        if is_validation:
            # [25, 10] is the neighbors to keep at each hop defined in the original paper
            validation_batches = NeighborLoader(ds, num_batch_neighbors, batch_size=batch_size, input_nodes=ds.val_mask,
                                                shuffle=False)
        else:
            validation_batches = NeighborLoader(ds, num_batch_neighbors, batch_size=batch_size,
                                                input_nodes=ds.test_mask, shuffle=False)
    else:
        validation_batches = [ds]
        if is_validation:
            mask = ds.val_mask
        else:
            mask = ds.test_mask

    eval_loss, eval_acc = .0, .0
    batch_num = 0
    for batch in validation_batches:
        batch = batch.to(device)
        # Count the number of nodes in the current batch
        if batch_generation:
            if is_validation:
                mask = batch.val_mask
            else:
                mask = batch.test_mask

        # Compute the response of the model
        out = model(batch.x, batch.edge_index)

        batch_num += torch.sum(
            mask).item()  # Number of values set to 1 in the mask: they are the samples used for computing the accuracy

        # Compute the loss function and with item() get the numerical value associated to it
        eval_loss += loss_fn(out[mask], batch.y[mask]).item()
        # Compute the argmax over the outputs (you may apply softmax as well, but it's just a scaling)
        cls = out.argmax(dim=-1)
        # Compute the number of correctly classified samples
        eval_acc += (torch.sum(cls[mask].eq(batch.y[mask]))).item()

    eval_loss /= batch_num
    eval_acc /= batch_num

    return eval_loss, eval_acc
