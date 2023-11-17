from torch_geometric.datasets import Planetoid

def load_ds(dataset_name, transform):
    dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=transform)

    return dataset

def print_ds_info(ds : Planetoid):
    print(f'Dataset {ds}')
    print('======================')
    print(f'Number of graphs: {len(ds)}')
    print(f'Number of features: {ds.num_features}')
    print(f'Number of classes: {ds.num_classes}')
    print('======================')
    data =  ds[0]
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')