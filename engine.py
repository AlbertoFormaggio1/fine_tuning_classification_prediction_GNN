from typing import List

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.sampler import NegativeSampling
from torch_geometric.utils import negative_sampling
from tqdm.auto import tqdm

import model


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

            if writer_info['second_tr_e'] != None:
                ep_to_add = writer_info['second_tr_e']
            else:
                ep_to_add = 0

            for k in results.keys():
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{writer_info["training_step"]}/{k}',
                                  results[k][-1], epoch + ep_to_add)
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{k}', results[k][-1],
                                  epoch + writer_info["starting_epoch"])

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
    model.train()  # Set the model in training mode
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


def train_link_prediction(model, train_ds, val_ds, loss_fn: torch.nn.Module,
                          opt: torch.optim.Optimizer, epochs: int, writer, writer_info, device, batch_generation: bool = False,
                          num_batch_neighbors: List[int] = [25, 10], batch_size: int = 32,
                          lr_schedule: torch.optim.lr_scheduler.LRScheduler = None):
    """
    This function trains a model to perform the task of link prediction.
    Link prediction is the task of, given 2 nodes, say whether it is likely to have an edge between those two nodes.

    :param model: the model to train on the task
    :param train_ds: graph G=(V,E') made of the nodes from the original graph V and a subset of the edges E' c E over
    which to perform the training task
    :param val_ds: graph G=(V,E') made of the nodes from the original graph V and a subset of the edges E' c E to use as
    validation set
    :param loss_fn: A PyTorch loss function to calculate loss on both datasets.
    :param opt: A PyTorch optimizer to help minimize the loss function.
    :param epochs: number of epochs to train the model
    :param writer: Tensorboard writer for writing the scalars generated during training
    :param writer_info: information on where to put the scalars generated during training
    :param device: the device on which to run inference. 'cuda' or 'cpu'
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

    # GraphSAGE works with generation of batches where you only pick a subset of the original graph
    # The other networks instead don't need that
    if batch_generation:
        # We keep the num_batch_neighbors[0] neighbors of each node and then num_batch_neighbors[1] neighbors for each of them
        # We do two hops
        train_batches = LinkNeighborLoader(train_ds, num_neighbors=num_batch_neighbors, neg_sampling=NegativeSampling('binary'),
                                           batch_size=batch_size,
                                           edge_label_index=train_ds.edge_label_index,
                                           edge_label=train_ds.edge_label,
                                           shuffle=True)

        correct_samples = train_loss = 0
        samples_num = 0
        for epoch in tqdm(range(epochs)):
            for batch in train_batches:
                batch = batch.to(device)  # Move the batch to the correct device
                # Train on the current batch
                loss_tmp, acc_tmp, bs = train_step_link_pred_batch_gen(model, batch, loss_fn, opt)

                # increase the metrics.
                correct_samples += acc_tmp
                train_loss += loss_tmp
                samples_num += bs

            # Get the train accuracy and train loss of the current epoch.
            # We divide by the number of samples the loss because we set the loss mode to "sum" => the loss for each sample is summed
            # We divide by the number of samples the accuracy since we return the number of correctly classified samples
            train_acc = correct_samples / samples_num
            train_loss /= samples_num

            # Updating the learning rate before doing validation
            if lr_schedule is not None:
                lr_schedule.step()

            # Get the results on the validation set
            val_loss, val_acc = eval_predictor(model, loss_fn, val_ds, batch_generation, device, num_batch_neighbors,
                                               batch_size)

            # Add the validation and training metrics
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)

            if writer_info['second_tr_e'] != None:
                ep_to_add = writer_info['second_tr_e']
            else:
                ep_to_add = 0

            # Add the metrics to the tensorboard writer.
            # We keep a chart cumulative of the whole training and one for each training stage (e.g. classification1 / link pred/  classification2)
            for k in results.keys():
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{writer_info["training_step"]}/{k}',
                                  results[k][-1], epoch + ep_to_add)
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{k}', results[k][-1],
                                  epoch + writer_info["starting_epoch"])

    else:
        for epoch in tqdm(range(epochs)):
            # Moving the training set to the right device
            train_ds = train_ds.to(device)

            train_loss, train_acc = train_step_link_pred_neg_sampling(model, train_ds, loss_fn, opt)

            val_loss, val_acc = eval_predictor(model, loss_fn, val_ds, batch_generation, device, num_batch_neighbors,
                                               batch_size)

            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)

            if writer_info['second_tr_e'] != None:
                ep_to_add = writer_info['second_tr_e']
            else:
                ep_to_add = 0

            for k in results.keys():
                writer.add_scalar(
                    f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{writer_info["training_step"]}/{k}',
                    results[k][-1], epoch + ep_to_add)
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{k}', results[k][-1],
                                  epoch + writer_info["starting_epoch"])

    # dovrebbe diventare "return results"
    return results #val_loss


def train_step_link_pred_batch_gen(model: torch.nn.Module, batch, loss_fn: torch.nn.Module,
                                   opt: torch.optim.Optimizer):
    """
    Performs a training step for the link prediction when batch generation is used to generate batches.
    In this case, the sampling must've been done before by using the LinkNeighborLoader to generate the batch

    :param model: the model to train
    :param batch: the batch to use to train
    :param loss_fn: A PyTorch loss function to calculate loss.
    :param opt: A PyTorch optimizer to help minimize the loss function.
    :return: (batch loss, # of correct samples,  batch size)
    """
    model.train()  # Set the model in training phase
    opt.zero_grad()
    # Computing first the embeddings with message passing on the edges that are already existing
    # in the graph
    z = model(batch.x, batch.edge_index)

    # Decode the predictions
    out = model.decode(z, batch.edge_label_index).view(-1)
    loss = loss_fn(out, batch.edge_label)

    batch_size = out.shape[0]
    # Compute the accuracy: the prediction is obtained by rounding the probability score obtained with sigmoid
    # in the range [0,1] to the closest integer: 1 if there is an edge, 0 otherwise
    train_acc = (torch.sum(batch.edge_label == torch.round(out.sigmoid()))).item()
    train_loss = loss.item()

    # Back-propagate the loss and update the parameters
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()

    return train_loss, train_acc, batch_size


def train_step_link_pred_neg_sampling(model: torch.nn.Module, train_ds, loss_fn: torch.nn.Module,
                                      opt: torch.optim.Optimizer):
    """
    A single training step for the link prediction task performing negative sampling over the full input dataset.
    Given a training graph G = (V, E), the edges in E are used as positive examples and |E| edges are generated randomly
    between pairs of nodes not sharing a link. Overall the training set of edges is made of 2|E| edges.

    :param model: the model to train
    :param train_ds: the graph containing the nodes and the edges in the training set
    :param loss_fn: A PyTorch loss function to calculate loss.
    :param opt: A PyTorch optimizer to help minimize the loss function.
    :return: train loss, # of correctly identified samples
    """
    model.train()  # Set the model in training phase
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

    out = model.decode(z, edge_label_index).view(-1)
    loss = loss_fn(out, edge_label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()

    train_acc = (torch.sum(edge_label == torch.round(out.sigmoid()))).item() / edge_label.shape[0]
    train_loss = loss.item() / edge_label.shape[0]

    return train_loss, train_acc


@torch.no_grad()
def eval_predictor(model: model.LinkPredictor, loss_fn, ds, batch_generation: bool, device,
                   num_batch_neighbors: List = [25, 10],
                   batch_size: int = 32):
    """
    Tests the model over the link prediction task by computing predicition accuracy and loss.

    :param model: model to test
    :param loss_fn: the criterion
    :param ds: the full graph dataset
    :param batch_generation: Whether batches with Neighbor Sampling should be generated (usually True for GraphSAGE)
    :param device: the device on which to run inference. 'cuda' or 'cpu'
    :param num_batch_neighbors: List containing how many nodes to sample in the NeighborSampling at each hop.
    Example: [25, 10] for each node, it will sample 25 neighbors and for each neighbor sampled, it will sample in turn
    10 neighbors.
    :param batch_size: how many sampled subgraphs for each batch
    :return: (loss, # samples correctly predicted)
    """
    model = model.to(device)
    model.eval()  # Set the model in evaluation stage

    # Generate the test batches if using SAGE
    if batch_generation:
        val_batches = LinkNeighborLoader(data=ds, num_neighbors=num_batch_neighbors,
                                         edge_label_index=ds.edge_label_index,
                                         edge_label=ds.edge_label, batch_size=batch_size, shuffle=False)
    else:
        val_batches = [ds]

    val_acc, val_loss = .0, .0
    batch_num = 0
    for batch in val_batches:
        batch = batch.to(device)
        z = model(batch.x, batch.edge_index)  # Compute the embeddings
        # Perform the decoding, flatten it by using view(-1) and then compute the confidence with the sigmoid activation function
        out = model.decode(z, batch.edge_label_index).view(-1).sigmoid()
        # Accuracy for link prediction
        val_acc += (torch.sum(batch.edge_label == torch.round(out))).item()
        val_loss += loss_fn(out, batch.edge_label).item()
        batch_num += out.shape[0]

    val_acc /= batch_num
    val_loss /= batch_num

    # Compute the accuracy score
    return val_loss, val_acc


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
