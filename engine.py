import torch
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.sampler import NegativeSampling
from torch_geometric.utils import negative_sampling
from tqdm.auto import tqdm
from typing import List

import model


def train_classification(model, train_ds, val_ds, loss_fn: torch.nn.Module,
          opt: torch.optim.Optimizer, epochs: int, writer, writer_info, device, batch_generation: bool = False, num_batch_neighbors: List[int] = [25, 10]):
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
        train_batches = NeighborLoader(train_ds, num_neighbors=num_batch_neighbors, batch_size=128, input_nodes=train_ds.train_mask,
                                       shuffle=True)
    else:
        train_batches = [train_ds]

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = .0, .0
        batch_num = 0
        for batch in train_batches:
            batch = batch.to(device)
            cur_train_loss, cur_train_acc = train_step(model=model, ds=batch, loss_fn=loss_fn, opt=opt)
            if batch_generation:
                # If we generated batches, than we get the size by doing batch.batch_size
                batch_size = batch.batch_size
            else:
                # Otherwise, we sum the values to 1 in the training mask (training mask has a 1 for every node to consider in the training set)
                batch_size = torch.sum(batch.train_mask).item()
            batch_num += batch_size
            # Increase the loss and the accuracy propotionally to the size of the batch
            train_loss += cur_train_loss * float(batch_size)
            train_acc += cur_train_acc * float(batch_size)

        # Compute the average loss and accuracy
        train_loss /= batch_num
        train_acc /= batch_num

        # results['train_loss'].append(train_loss)
        # results['train_acc'].append(train_acc)

        # every 10 epochs see the improvement on the validation set
        if epoch % 5 == 0 or epoch == epochs-1:
            val_loss, val_acc = eval_classifier(model, loss_fn, val_ds, True, batch_generation, device)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)

            for k in results.keys():
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{writer_info["training_step"]}/{k}',
                                  results[k][-1], epoch)
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{k}', results[k][-1],
                                  epoch + writer_info["starting_epoch"])

    return results

def train_step(model: torch.nn.Module, ds, loss_fn: torch.nn.Module,
               opt: torch.optim.Optimizer):
    model.train()  # Set the model in training mode
    opt.zero_grad()  # Reset the gradient
    out = model(ds.x, ds.edge_index)  # Compute the response of the model
    loss = loss_fn(out[ds.train_mask], ds.y[ds.train_mask])  # Compute the loss based on training nodes
    loss.backward()  # Propagate the gradient
    opt.step()  # Update the weights

    train_loss = loss.item()  # Get the loss
    # Compute the classification accuracy
    train_cls = out.argmax(dim=-1)
    train_acc = torch.sum(train_cls[ds.train_mask] == ds.y[ds.train_mask]) / torch.sum(ds.train_mask)

    return train_loss, train_acc.item()


def train_link_prediction(model, train_ds, val_ds, loss_fn: torch.nn.Module,
                          opt: torch.optim.Optimizer, epochs: int, writer, writer_info, device, batch_generation: bool = False,
                          num_batch_neighbors: List[int] = [25, 10]):
    """
    This function trains a link predictor model
    """
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }

    # GraphSAGE works with generation of batches where you only pick a subset of the original graph
    # The other networks instead don't need that
    if batch_generation:
        # We keep the 25 neighbors of each node and then 10 neighbors for each of them
        # They trained on 10 epochs for the fully supervised sampling
        train_batches = LinkNeighborLoader(train_ds, num_neighbors=num_batch_neighbors, neg_sampling=NegativeSampling('binary'),
                                           batch_size=128,
                                           edge_label_index=train_ds.edge_label_index,
                                           edge_label=train_ds.edge_label,
                                           shuffle=True)

        train_acc = train_loss = 0
        samples_num = 0
        for epoch in tqdm(range(epochs)):
            for batch in train_batches:
                batch = batch.to(device)
                model.train()  # Set the model in training phase
                opt.zero_grad()
                # Computing first the embeddings with message passing on the edges that are already existing
                # in the graph

                z = model(batch.x, batch.edge_index)

                out = model.decode(z, batch.edge_label_index).view(-1)
                loss = loss_fn(out, batch.edge_label)

                train_acc += (torch.sum(batch.edge_label == torch.round(out.sigmoid()))).item() * float(out.shape[0])
                train_loss += loss.item() * float(out.shape[0])
                samples_num += out.shape[0]

                loss.backward()
                opt.step()

            train_acc /= samples_num
            train_loss /= samples_num

            val_loss, val_acc = eval_predictor(model, loss_fn, val_ds, batch_generation, device)

            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)

            for k in results.keys():
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{writer_info["training_step"]}/{k}',
                                  results[k][-1], epoch)
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{k}', results[k][-1],
                                  epoch + writer_info["starting_epoch"])

    else:
        for epoch in tqdm(range(epochs)):
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
            opt.step()

            train_acc = (torch.sum(edge_label == torch.round(out.sigmoid()))).item() / edge_label.shape[0]
            train_loss = loss.item()

            val_loss, val_acc = eval_predictor(model, loss_fn, val_ds, batch_generation)

            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)

            for k in results.keys():
                writer.add_scalar(
                    f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{writer_info["training_step"]}/{k}',
                    results[k][-1], epoch)
                writer.add_scalar(f'{writer_info["dataset_name"]}/{writer_info["model_name"]}/{k}', results[k][-1],
                                  epoch + writer_info["starting_epoch"])

    # dovrebbe diventare "return results"
    return val_loss


@torch.no_grad()
def eval_predictor(model: model.LinkPredictor, loss_fn, ds, batch_generation: bool, device):
    model.eval()  # Set the model in evaluation stage

    if batch_generation:
        val_batches = LinkNeighborLoader(data=ds, num_neighbors=[25, 10], edge_label_index=ds.edge_label_index,
                                         edge_label=ds.edge_label, batch_size=3 * 128, shuffle=False)
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
def eval_classifier(model, loss_fn, ds, is_validation, batch_generation, device):
    model.eval()

    if batch_generation:
        if is_validation:
            # [25, 10] is the neighbors to keep at each hop defined in the original paper
            validation_batches = NeighborLoader(ds, [25, 10], batch_size=128, input_nodes=ds.val_mask, shuffle=False)
        else:
            validation_batches = NeighborLoader(ds, [25, 10], batch_size=128, input_nodes=ds.test_mask, shuffle=False)
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
