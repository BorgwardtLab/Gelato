import torch
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def normalized_mae(predictions, labels):
    # Convert to numpy arrays for easy vectorized computation
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate the absolute error, normalized by the true value
    error = np.abs(predictions - labels) / np.maximum(1, labels)
    
    # Calculate the mean of the normalized errors
    normalized_error = np.mean(error)
    
    return normalized_error


def exact_hit_rate(predictions, labels):
    # Convert to numpy arrays for easy vectorized computation
    predictions = np.array(predictions)
    labels = np.array(labels)
    return (predictions == labels).mean()


def matching_cost(g1, g2, matching, node_cost, edge_cost):
    device = g1.x.device

    # node ops
    mask = matching >= 0
    g1_feats = g1.x[mask]
    g2_feats = g2.x[matching[mask]]  
    node_rel_ops = (g1_feats != g2_feats).any(dim=1).sum().item()
    node_del_ops = g1.x.size(0) - g1_feats.size(0)
    node_ins_ops = g2.x.size(0) - g2_feats.size(0)

    # edge ops
    n = g2.x.size(0)
    adjm = torch.zeros((n, n,), dtype=bool, device=device) # (n,n)
    attr = torch.zeros((n, n, g2.edge_attr.size(1)), dtype=g2.edge_attr.dtype, device=device) # (n,n,d)
    idx = g2.edge_index[0] * n + g2.edge_index[1]
    adjm.view(-1)[idx] = True
    attr.view(n * n, -1)[idx] = g2.edge_attr
    matched_index = matching[g1.edge_index]
    adjm = adjm[matched_index[0], matched_index[1]] # (E,)
    attr = attr[matched_index[0], matched_index[1]] # (E, d)
    attr = attr[adjm] # (E_match, d)

    num_adj_matches = adjm.sum().item()
    edge_del_ops = (g1.edge_index.size(1) - num_adj_matches) // 2
    edge_ins_ops = (g2.edge_index.size(1) - num_adj_matches) // 2
    edge_rel_ops = ( g1.edge_attr[adjm] != attr ).any(dim=1).sum().item() // 2

    # print(node_del_ops, node_ins_ops, node_rel_ops, edge_del_ops, edge_ins_ops, edge_rel_ops)
    return node_cost*(node_del_ops+node_ins_ops+node_rel_ops)+edge_cost*(edge_del_ops+edge_ins_ops+edge_rel_ops)

def ensemble_matching_cost(g1, g2, matchings, node_cost, edge_cost):
    device = g1.x.device

    B = matchings.size(0)
    n1 = g1.x.size(0)
    n2 = g2.x.size(0)
    
    mask = matchings >= 0 # (B, n1)
    matchings_clamped = matchings.clamp(min=0)


    # -------- Node operations (vectorized over batch) --------
    g1_x_exp = g1.x.unsqueeze(0).expand(B, -1, -1) # (B, n1, d_node)
    g2_x_mapped = g2.x[matchings_clamped]

    node_diff_any = (g1_x_exp != g2_x_mapped).any(dim=2) # (B, n1)
    node_rel_ops = (node_diff_any & mask).sum(dim=1) # (B,)
    node_del_ops = (~mask).sum(dim=1) # (B,) nodes in g1 not matched
    node_ins_ops = n2 - mask.sum(dim=1) # (B,) per-row matched count -> inserts


    # -------- Edge operations (vectorized over batch) --------
    E1 = g1.edge_index.size(1)

    # build dense adjacency + attr representation for g2 (shape n2 x n2)
    adjm = torch.zeros((n2, n2), dtype=torch.bool, device=device)
    attr = torch.zeros((n2, n2, g2.edge_attr.size(1)), dtype=g2.edge_attr.dtype, device=device)


    idx = g2.edge_index[0] * n2 + g2.edge_index[1]
    adjm.view(-1)[idx] = True
    attr.view(n2 * n2, -1)[idx] = g2.edge_attr

    # For each g1 edge (u->v) get the mapped indices in g2 for each batch
    e_u = g1.edge_index[0]
    e_v = g1.edge_index[1]

    matched_u = matchings_clamped[:, e_u] # (B, E1)
    matched_v = matchings_clamped[:, e_v] # (B, E1)
    valid_edge_mask = (mask[:, e_u] & mask[:, e_v]) # (B, E1)
    flat_idx = matched_u * n2 + matched_v # (B, E1)

    adjm_flat = adjm.view(-1) # (n2*n2,)
    attr_flat = attr.view(n2 * n2, -1) # (n2*n2, d_edge)

    # whether corresponding g2 edge exists for each (batch, edge)
    adj_exists = adjm_flat[flat_idx] & valid_edge_mask # (B, E1)
    num_adj_matches = adj_exists.sum(dim=1) # (B,)

    edge_del_ops = (E1 - num_adj_matches) // 2
    edge_ins_ops = (g2.edge_index.size(1) - num_adj_matches) // 2


    g2_edge_attrs_mapped = attr_flat[flat_idx]
    g1_edge_attr_exp = g1.edge_attr.unsqueeze(0).expand(B, -1, -1)

    edge_diff_any = (g1_edge_attr_exp != g2_edge_attrs_mapped).any(dim=2) # (B, E1)
    edge_rel_ops = ((edge_diff_any) & adj_exists).sum(dim=1) // 2 # (B,)


    # -------- Combine costs --------
    costs = (
    node_cost * (node_del_ops + node_ins_ops + node_rel_ops).to(torch.float)
    + edge_cost * (edge_del_ops + edge_ins_ops + edge_rel_ops).to(torch.float)
    )


    return costs





def run_inference(model, dataset, k, batch_size=32, disable_tqdm=False):
    nc, ec = model.nc, model.ec
    g1L, g2L, full_mL = [], [], []
    costs, true_costs = [], []

    for data in tqdm(dataset, disable=disable_tqdm):
        g1 = Data(x=data.x_s, edge_index=data.edge_index_s, edge_attr=data.edge_attr_s)
        g2 = Data(x=data.x_t, edge_index=data.edge_index_t, edge_attr=data.edge_attr_t)
        g1L.append(g1)
        g2L.append(g2)
        full_mL.append(data.matching.long())

        if len(g1L) >= batch_size:
            matchingL = model.batch_ensemble_inference(g1L, g2L, k=k)
            for b in range(len(g1L)):
                costs.append( matching_cost(g1L[b], g2L[b], matchingL[b], node_cost=nc, edge_cost=ec) )
                true_costs.append( matching_cost(g1L[b], g2L[b], full_mL[b], node_cost=nc, edge_cost=ec) )
            g1L, g2L, full_mL = [], [], []     

    if len(g1L) > 0:
        matchingL = model.batch_ensemble_inference(g1L, g2L)
        for b in range(len(g1L)):
            costs.append( matching_cost(g1L[b], g2L[b], matchingL[b], node_cost=nc, edge_cost=ec) )
            true_costs.append( matching_cost(g1L[b], g2L[b], full_mL[b], node_cost=nc, edge_cost=ec) )
    
    return costs, true_costs

def training_step_link(model, loader, optimizer, args):
    epoch_loss = 0.0
    epoch_acc = 0.0
    cnt = 0
    model.train()
    cntp = 0
    cntn = 0
    for ii, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(args.device)
        
        logits = model(data)
        y_hat = (torch.sigmoid(logits) > 0.5).long()

        num_pos = (data.edge_label == 1).sum()
        num_neg = (data.edge_label == 0).sum()
        pos_weight = num_neg / num_pos
        
        loss = F.binary_cross_entropy_with_logits(logits, data.edge_label.float(), pos_weight=pos_weight) 
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        epoch_acc += ( data.edge_label == y_hat ).sum().detach().item() 
        cnt += data.edge_label.shape[0]

        cntp += data.edge_label.sum().item()
        cntn += (data.edge_label == 0).sum().item()

        if ii >= args.max_train_steps * len(loader):
            break

    print( cntp, cntn, cnt, len(loader) )

    epoch_loss /= len(loader)
    epoch_acc /= cnt
    return epoch_loss, epoch_acc

def validation_step_link(model, loader, args):
    epoch_loss = 0.0
    epoch_acc = 0.0
    cnt = 0
    model.eval()
    for data in loader:
        data = data.to(args.device)

        logits = model(data)
        y_hat = (torch.sigmoid(logits) > 0.5).long()

        num_pos = (data.edge_label == 1).sum()
        num_neg = (data.edge_label == 0).sum()
        pos_weight = (num_neg / num_pos) * 0.2
        
        loss = F.binary_cross_entropy_with_logits(logits, data.edge_label.float(), pos_weight=pos_weight) 

        epoch_loss += loss.detach().item()
        epoch_acc += ( data.edge_label == y_hat ).sum().detach().item() 
        cnt += data.edge_label.shape[0]


    epoch_loss /= len(loader)
    epoch_acc /= cnt
    return epoch_loss, epoch_acc