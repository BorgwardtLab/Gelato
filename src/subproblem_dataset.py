from torch_geometric.data import Data
import torch.nn.functional as F
import torch
from torch_scatter import scatter_max
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import gdown, zipfile

from .wl import WeisfeilerLeman
from .dataset import GraphMatchingDataset





def make_instance(d1, d2, pm, delete_e=True):
    n_s = d1.x.shape[0]
    n_t = d2.x.shape[0]
            
    # merge graphs and add dummy 
    dummy = torch.zeros((1, d1.x.shape[1]), device=d1.x.device)
    x = torch.cat([d1.x, d2.x, dummy], dim=0)
    edge_index = torch.cat([d1.edge_index, d2.edge_index + n_s], dim=1)
    edge_attr = torch.cat([d1.edge_attr, d2.edge_attr], dim=0)
    
    # mark source and target nodes
    m_feat = torch.zeros((x.shape[0], 2), device=x.device)
    m_feat[:n_s, 0] = 1
    m_feat[n_s:-1, 1] = 1
    x = torch.cat([x, m_feat], dim=1)

    # create new edges from matching
    s_idx = torch.nonzero(pm >= 0, as_tuple=True)[0] # matching source nodes
    t_idx = pm[pm >= 0] + n_s  # matching target nodes
    m = s_idx.shape[0]

    edge_attr = torch.cat([edge_attr, torch.zeros((edge_attr.shape[0], 1), device=edge_attr.device)], dim=1) # add new dimension
    new_edge_index = torch.zeros((2, 2*m), device=edge_index.device, dtype=torch.long)
    new_edge_attr = torch.zeros((2*m, edge_attr.shape[1]), device=edge_index.device)
    new_edge_attr[:, -1] = 1
    new_edge_index[:, :m] = torch.stack([s_idx, t_idx], dim=0)
    new_edge_index[:, m:] = torch.stack([t_idx, s_idx], dim=0)
    edge_index = torch.cat( [edge_index, new_edge_index], dim=-1 )
    edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)

    if not delete_e:
        return x, edge_index, edge_attr

    # compute disabled nodes
    unmatched_s = (pm < 0)
    unmatched_t = torch.ones(n_t, dtype=bool, device=x.device)
    unmatched_t[ pm[pm>=0] ] = False

    src, dst = d1.edge_index
    src_s = torch.cat( [src, torch.arange(n_s, device=x.device)] )
    dst_s = torch.cat( [dst, torch.arange(n_s, device=x.device)] )
    src, dst = d2.edge_index
    src_t = torch.cat( [src, torch.arange(n_t, device=x.device)] )
    dst_t = torch.cat( [dst, torch.arange(n_t, device=x.device)] )
    
    #print(s_idx, t_idx, x.shape[0])
    has_unmatched_neigh_s = scatter_max( unmatched_s[dst_s].to(torch.long) , src_s, dim=0, dim_size=x.shape[0])[0]
    has_unmatched_neigh_t = scatter_max( unmatched_t[dst_t].to(torch.long) , src_t + n_s, dim=0, dim_size=x.shape[0])[0]
    has_unmatched_neigh = torch.maximum(has_unmatched_neigh_s, has_unmatched_neigh_t).to(torch.bool) 

    
    self_and_match_have_unmatched_neigh = (has_unmatched_neigh[s_idx] & has_unmatched_neigh[t_idx])
    mask = torch.cat((unmatched_s, unmatched_t), dim=0) # now, keep just unmatched
    mask[s_idx] = self_and_match_have_unmatched_neigh # add source nodes with an unmatched neighbor and whose match has an unmatched neighbor
    mask[t_idx] = self_and_match_have_unmatched_neigh # add target nodes with an unmatched neighbor and whose match has an unmatched neighbor

    # delete edges with disabled nodes
    src, dst = edge_index
    keep = mask[src] & mask[dst]  
    edge_index = edge_index[:, keep]
    edge_attr = edge_attr[keep]

    return x, edge_index, edge_attr




def parallel_process(args):
    graph_pair, pm, num_node_labels, num_edge_labels = args
    full_m = graph_pair.matching.long()
    pm = pm.long()
    n_s, n_t = graph_pair.x_s.shape[0], graph_pair.x_t.shape[0]

    x_s = F.one_hot(graph_pair.x_s.view(-1).long(), num_node_labels)
    x_t = F.one_hot(graph_pair.x_t.view(-1).long(), num_node_labels)
    edge_index_s = graph_pair.edge_index_s.long()
    edge_index_t = graph_pair.edge_index_t.long()
    edge_attr_s = F.one_hot(graph_pair.edge_attr_s.view(-1).long(), num_edge_labels)
    edge_attr_t = F.one_hot(graph_pair.edge_attr_t.view(-1).long(), num_edge_labels)

    g1 = Data(x=x_s, edge_index=edge_index_s, edge_attr=edge_attr_s)
    g2 = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t)

    x, edge_index, edge_attr = make_instance(g1, g2, pm, delete_e=True)

    # get RWPE to make WL stronger
    adj = torch.zeros((x.shape[0], x.shape[0]), device=x.device)
    adj[edge_index[0], edge_index[1]] = 1
    loop_index = torch.arange(x.shape[0], device=x.device)
    out = adj
    pe = [ out[loop_index, loop_index] ]
    for _ in range(5):
        out = out @ adj
        pe.append( out[loop_index, loop_index] )
    pe = torch.stack(pe, dim=-1)
    x_pe = torch.cat([x, pe], dim=-1)

    # now run WL to get node orbits
    wl = WeisfeilerLeman(5)
    wl_labels = wl.run(x=x_pe, edge_index=edge_index, edge_attr=edge_attr)



    # # prepare link preiction labels
    unmatched_nodes_s = torch.nonzero(pm == -2, as_tuple=False).view(-1)
    unmatched_t = torch.ones(n_t, dtype=bool, device=x.device)
    unmatched_t[ pm[pm>=0] ] = False
    unmatched_nodes_t = torch.nonzero(unmatched_t, as_tuple=False).view(-1)

    u = full_m[unmatched_nodes_s]
    wl_s = wl_labels[unmatched_nodes_s] 
    wl_t = wl_labels[u + n_s]
    edge_label_index_pos = torch.stack([unmatched_nodes_s, u+n_s], dim=1)
    edge_classes = torch.unique(wl_s+wl_t)
    
    edge_label_index_neg = []
    for next_s in unmatched_nodes_s:
        wl_s = wl_labels[next_s]
        wl_u = wl_labels[unmatched_nodes_t + n_s]
        edge_class = wl_s+wl_u
        mask = ~( (edge_class.view(1,-1) == edge_classes.view(-1,1)).any(dim=0) )
        u_idxs = unmatched_nodes_t[ torch.nonzero(mask, as_tuple=True)[0] ]
        edges = torch.stack([torch.full_like(u_idxs, next_s), u_idxs + n_s], dim=1)
        edge_label_index_neg.append(edges)
    edge_label_index_neg = torch.cat(edge_label_index_neg, dim=0)

    edge_label_index = torch.cat([edge_label_index_pos, edge_label_index_neg], dim=0).t()
    edge_label = torch.cat([
        torch.ones(edge_label_index_pos.size(0), dtype=torch.int8, device=edge_label_index_pos.device),
        torch.zeros(edge_label_index_neg.size(0), dtype=torch.int8, device=edge_label_index_neg.device)
    ], dim=0)
    

    data = Data(
        x=x.to(torch.int8), 
        edge_index=edge_index.to(torch.int32), 
        edge_attr=edge_attr.to(torch.int8), 
        edge_label_index=edge_label_index.to(torch.int32), 
        edge_label=edge_label.to(torch.int8)
    ) 
    return data




class GraphMatchingSubproblemDataset(GraphMatchingDataset):
    r"""A dataset class for graph matching sub-instances.
    """
    def __init__(self, *, num_instances_per_pair=None, parallel=True, **kwargs):
        self.num_instances_per_pair = num_instances_per_pair
        self.parallel = parallel
        super().__init__(**kwargs)
    
    
    @property
    def processed_file_names(self):
        if self.bounds is not None: return [f'data_{self.split}_{self.bounds[0]}-{self.bounds[1]}_{self.num_pairs}_{self.num_instances_per_pair}_{self.seed}.pt']
        else: return [f'data_{self.split}_{self.num_pairs}_{self.num_instances_per_pair}_{self.seed}.pt']
    
    
    def get_data_list(self, data_list, num_node_labels, num_edge_labels):
        instance_tuples = []
        for i, data in enumerate(data_list):
            g1_size = data.x_s.size(0)
            instance_tuples.append((i, 0))
            for _ in range(self.num_instances_per_pair-1):
                target_size = random.randint(1, g1_size-1)
                instance_tuples.append((i, target_size))

        args_list = []
        for idx in range(len(instance_tuples)):  
            i, target_size = instance_tuples[idx]
            graph_pair = data_list[i]
            
            n_s = graph_pair.x_s.size(0)
            partial_matching = torch.full((n_s,), -2, dtype=torch.int16)
            Slist = torch.randperm(n_s)[:target_size]
            partial_matching[Slist] = graph_pair.matching[Slist]

            args_list.append( 
                (graph_pair, partial_matching, num_node_labels, num_edge_labels)
                )

        if self.parallel:
            data_list = Parallel(n_jobs=-2)(delayed(parallel_process)(args) for args in tqdm(args_list))
        else:
            data_list = [ parallel_process(args) for args in tqdm(args_list) ]
        return data_list
