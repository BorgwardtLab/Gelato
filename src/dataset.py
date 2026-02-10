import os
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric
import torch.nn.functional as F
import torch
import random
from tqdm import tqdm
import json
import gdown, zipfile





def load_graphs(path, nomap=False):
    node_attrs = set()
    edge_attrs = set()

    with open(path) as f:
        lines = f.read().splitlines()

    graphs = [] 
    tmp = []   
    num_graphs = int(lines[0])
    idx = 1

    for _ in range(num_graphs):
        node_attr = lines[idx+1].split()
        node_attr = list(map(int, node_attr))
        for xx in node_attr: node_attrs.add(xx)
        
        idx += 2

        edge_index, edge_attr = [], []
        while idx < len(lines) and not lines[idx].isdigit():
            u, v, a = map(int, lines[idx].split())
            edge_index.append([u, v])
            edge_attr.append(a)
            idx += 1
        for xx in edge_attr: edge_attrs.add(xx)

        tmp.append((node_attr, edge_index, edge_attr))
    
    node_attrs = sorted(list(node_attrs))
    edge_attrs = sorted(list(edge_attrs))
    if nomap:
        node2id = {i: i for i in range(120)}
        edge2id = {i: i for i in range(6)}
    else:
        node2id = {attr: i for i, attr in enumerate(node_attrs)}
        edge2id = {attr: i for i, attr in enumerate(edge_attrs)}

    for x, edge_index, edge_attr in tmp:
        x = torch.tensor( [node2id[attr] for attr in x] , dtype=torch.long).unsqueeze(1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([edge2id[attr] for attr in edge_attr], dtype=torch.long)
        edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index, edge_attr, reduce='max')

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    return graphs, node2id, edge2id


def load_graph_matchings(path):
    with open(path) as f:
        lines = f.read().splitlines()

    maps = {}
    idx = 0

    while idx < len(lines):
        if not lines[idx] or lines[idx][0] == '#':
            idx += 1
            continue

        parts = lines[idx].split()
        g1, g2, n, ged_max, ged_min = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
        idx += 1

        matching = torch.empty(n, dtype=torch.int8)
        for i in range(n):
            u = int(lines[idx+i])
            matching[i] = u

        maps[(g1, g2)] = matching
        idx += n

    return maps




def reorder_graph_pairs(graphs, pair2map, order='large_to_small'):
    assert order in ['large_to_small', 'small_to_large']
    new_maps = {}
    for (g1,g2), mapp in pair2map.items():
        g1_size, g2_size = graphs[g1].x.size(0), graphs[g2].x.size(0)
        if (order=='large_to_small' and g1_size < g2_size) or (order=='small_to_large' and g1_size >= g2_size):
            reversed_map = torch.full((g2_size,), -1, device=mapp.device)
            valid = mapp >= 0
            reversed_map[mapp[valid].long()] = torch.arange(g1_size, device=mapp.device)[valid]
            new_maps[(g2,g1)] = reversed_map
        else:
            new_maps[(g1,g2)] = mapp
    return new_maps







class GraphMatchingDataset(InMemoryDataset):
    r"""A dataset class for graph matching instances.
    """

    urls = {
        'aids':     '1t4JQBiaSBqknNimNU4nvQ3vHKJEWCi0-', 
        'linux':    '1Ro8KTSz4G3pFweD75SEVZ3kI3jPkubGx',
        'imdb-16':  '14vKkIahcHMPPDh1BJf7u4-KxAhD9uqse', 
        'zinc-16':  '1Po-wwBienIFL6n_RH8Z_csZsB-S1XxP6', 
        'molhiv-16':'1zJR3eIOm0WkJwpY0gs3aktnHTcdqwWi7', 
        'code2-22': '11uqVWicdPCpYkAbPJ6Aug9TPDxWIFYTw',
        'molhiv-16-edge-unlabeled': '1eNwonvtPkMLLPtAQiy_GsWw_jTzLHo3k',
        'zinc-16-edge-unlabeled':   '1GWts3DTAjsInZou_hZJkJM1o2fXPQqAx',
    }

    def __init__(self, root='data/', name='aids', num_pairs=None, split=None, seed=0, bounds=None, transform=None):
        self.name = name
        self.num_pairs = num_pairs
        self.seed = seed
        self.split = split
        self.bounds = bounds
        assert split in ['train', 'val', 'test', 'larger', 'all']
        assert (bounds is None) or (len(bounds) == 2)
        assert name in ['aids', 'linux', 'imdb-16', 'zinc-16', 'molhiv-16', 'code2-22', 'molhiv-16-edge-unlabeled', 'zinc-16-edge-unlabeled']

        super().__init__(root+name)
        self.load(self.processed_paths[0])
    

    @property
    def raw_file_names(self):
        return [f'{self.name}.zip']

    @property
    def processed_file_names(self):
        if self.bounds is not None: return [f'data_{self.split}_{self.bounds[0]}-{self.bounds[1]}_{self.num_pairs}_{self.seed}.pt']
        return [f'data_{self.split}_{self.num_pairs}_{self.seed}.pt']
    
    def download(self):
        url = f'https://drive.google.com/uc?id={self.urls[self.name]}'

        # Use gdown to handle the download
        output = os.path.join(self.raw_dir, self.raw_file_names[0])
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, 'r') as f:
            f.extractall(self.raw_dir)


    def process(self):
        random.seed(self.seed)

        graphs, node2id, edge2id = load_graphs(os.path.join(self.raw_dir, self.name, 'graphs.txt'), nomap=True)
        pair2map = load_graph_matchings(os.path.join(self.raw_dir, self.name, 'match.txt'))
        pair2map = reorder_graph_pairs(graphs, pair2map, order='small_to_large') # reorder data 

        num_node_labels, num_edge_labels = len(node2id), len(edge2id)
        if self.split != 'all':
            with open(os.path.join(self.raw_dir, self.name, 'splits.json'), "r") as f:
                all_indices = json.load(f)
            train_indices, val_indices, test_indices, larger_indices = all_indices

        # select all graph pairs within split
        if self.split == 'train': split_indices_set = set(train_indices)
        elif self.split == 'val': split_indices_set = set(val_indices)
        elif self.split == 'test': split_indices_set = set(test_indices)
        elif self.split == 'larger': split_indices_set = set(larger_indices)
        else: split_indices_set = set(range(len(graphs)))

        # filter graphs by size
        if self.bounds is not None: 
            split_indices_set = split_indices_set & {i for i, g in enumerate(graphs) if ( (g.x.shape[0] >= self.bounds[0]) and (g.x.shape[0] <= self.bounds[1]) )}

        # retain only pairs in split
        pairs = list(pair2map.keys())
        pairs = [(i, j) for i, j in pairs if i in split_indices_set and j in split_indices_set]
        random.shuffle(pairs)

        # select subset of pairs
        if self.num_pairs is not None and self.num_pairs < len(pairs):
            pairs = random.sample(pairs, self.num_pairs)

        # prepare data list
        data_list = []
        for idx in tqdm(range(len(pairs))):  
            i1, i2 = pairs[idx]
            g1, g2 = graphs[i1].clone(), graphs[i2].clone()
            matching = pair2map[(i1, i2)].clone().to(torch.int16)

            x_s = g1.x.view(-1).to(torch.int8)
            x_t = g2.x.view(-1).to(torch.int8)
            edge_index_s = g1.edge_index.to(torch.int)
            edge_index_t = g2.edge_index.to(torch.int)
            edge_attr_s = g1.edge_attr.view(-1).to(torch.int8)
            edge_attr_t = g2.edge_attr.view(-1).to(torch.int8)

            data_list.append( 
                Data(x_s=x_s, x_t=x_t, edge_index_s=edge_index_s, edge_index_t=edge_index_t, edge_attr_s=edge_attr_s, edge_attr_t=edge_attr_t, matching=matching, i1=i1, i2=i2) 
            )

        # transform the data list if necessary
        data_list = self.get_data_list(data_list, num_node_labels, num_edge_labels)

        self.save(data_list, self.processed_paths[0])
    

    def get_data_list(self, data_list, num_node_labels, num_edge_labels):
        for data in data_list:
            data.x_s = F.one_hot(data.x_s.view(-1).long(), num_node_labels)
            data.x_t = F.one_hot(data.x_t.view(-1).long(), num_node_labels)
            data.edge_attr_s = F.one_hot(data.edge_attr_s.view(-1).long(), num_edge_labels)
            data.edge_attr_t = F.one_hot(data.edge_attr_t.view(-1).long(), num_edge_labels)

        return data_list




