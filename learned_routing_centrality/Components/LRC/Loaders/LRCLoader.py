import os.path
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDatasetLRCLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, data_paths, dtype, device):
        self.data_paths = data_paths
        self.dtype = dtype
        self.device = device
        self.loaded_data = {}
        for i, current_path in enumerate(self.data_paths):
            num_nodes = torch.load(os.path.join(f'{current_path}', 'n_nodes.pt')).to(dtype=self.dtype)
            expected_rbc = torch.load(os.path.join(f'{current_path}', 'expected_rbc.pt')).to(dtype=self.dtype)
            expected_rbc = expected_rbc / torch.sum(expected_rbc)
            node_embeddings = torch.load(os.path.join(f'{current_path}', 'nodes_embeddings.pt')).to(dtype=self.dtype)
            edges = np.genfromtxt(os.path.join(f'{current_path}', 'edges.txt'), delimiter=',', dtype=int)
            self.loaded_data.update({i: (num_nodes, expected_rbc, node_embeddings, edges)})

    def __getitem__(self, index):
        return self.loaded_data[index]

    def __len__(self):
        return len(self.data_paths)


def custom_collate_fn(batch):
    num_nodes_lst = [int(batch[i][0].item()) for i in range(len(batch))]
    expected_rbcs_lst = [batch[i][1] for i in range(len(batch))]
    node_embeddings_lst = [batch[i][2] for i in range(len(batch))]
    edges_lst = [batch[i][3] for i in range(len(batch))]

    return (num_nodes_lst, node_embeddings_lst, edges_lst), expected_rbcs_lst
