import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric.utils import unbatch
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_max

from .utils import ensemble_matching_cost




class LinkGNN(nn.Module):
    def __init__(self, num_node_labels, num_edge_labels, hidden_dim, num_layers, nc, ec):
        super().__init__()
        self.num_node_labels = num_node_labels
        self.num_edge_labels = num_edge_labels
        self.hidden_dim = hidden_dim

        self.embed = nn.Linear(num_node_labels, hidden_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            nn_seq = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers.append(GINEConv(nn_seq, edge_dim=num_edge_labels))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.out_head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.nc = nc # node costs
        self.ec = ec # edge costs
        
    
    def forward(self, batch):
        x, edge_index, edge_attr = batch.x.float(), batch.edge_index.long(), batch.edge_attr.float()
        # apply GINE layers
        x = self.embed(x)
        for conv, bn in zip(self.layers, self.bns):
            x_upd = conv(x, edge_index, edge_attr)
            x_upd = bn(x_upd)
            x_upd = torch.relu(x_upd)
            x = x_upd + x
        
        src, dst = batch.edge_label_index.long()
        out = torch.cat([ x[src], x[dst] ], dim=1)
        out = self.out_head(out) # logits
        return out.view(-1)





    

    def batch_ensemble_inference(self, sources_, targets_, k=1):
        ddevice = next(self.parameters()).device
        B = len(sources_)
        n_sL, n_tL = [None for b in range(B)], [None for b in range(B)]
        matchingL, unmatched_sL, unmatched_tL = [None for b in range(B)], [None for b in range(B)], [None for b in range(B)]
        instanceL = [None for b in range(B)]
        for b in range(B):
            source = sources_[b].clone()
            target = targets_[b].clone()
            device = source.x.device
            n_s, n_t = source.x.shape[0], target.x.shape[0] # note that x.shape[0] = n_s + n_t + 1 
            n_sL[b] = n_s
            n_tL[b] = n_t
            matching = torch.full((k, n_s), -2, device=device)
            unmatched_s = torch.ones((k, n_s), dtype=bool, device=device)
            unmatched_t = torch.ones((k, n_t), dtype=bool, device=device)
            matchingL[b] = matching
            unmatched_sL[b] = unmatched_s
            unmatched_tL[b] = unmatched_t
            dummy = torch.zeros((1, source.x.shape[1]), device=device)
            x = torch.cat([source.x, target.x, dummy], dim=0).float()
            edge_index = torch.cat([source.edge_index, target.edge_index + n_s], dim=1).long()
            edge_attr = torch.cat([source.edge_attr, target.edge_attr], dim=0).float()
            edge_attr = torch.cat([edge_attr, torch.zeros((edge_attr.shape[0], 1), device=device)], dim=1) # add new dimension
            m_feat = torch.zeros(((n_s+n_t+1), 2), device=device)
            m_feat[:n_s, 0] = 1
            m_feat[n_s:-1, 1] = 1
            x = torch.cat([x, m_feat], dim=1)
            edge_label_index = torch.cartesian_prod( torch.arange(n_s, device=device) , torch.arange(n_t, device=device) + n_s ).t()
            instance = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_label_index=edge_label_index) 
            instanceL[b] = instance
        instance = Batch.from_data_list(instanceL, follow_batch=['edge_label_index'])
        preds = self.forward(instance.to(ddevice)).cpu()
        predsL = unbatch(preds, instance.edge_label_index_batch)
        batchs = []
        node_offsets = torch.zeros(B*k, device=device, dtype=int)
        prev_offset = 0   
        for b in range(B):
            n_s = n_sL[b]
            n_t = n_tL[b]
            matching = matchingL[b]
            unmatched_s = unmatched_sL[b] 
            unmatched_t = unmatched_tL[b]
            preds = predsL[b].clone()
            instance = instanceL[b]
            batchs.append( b*k + torch.arange(k, device=device).repeat_interleave((n_s+n_t+1)) )
            node_offsets[b*k:(b+1)*k] = prev_offset + (n_s+n_t+1)*torch.arange(k, device=device, dtype=int)
            prev_offset = node_offsets[(b+1)*k -1] + (n_s+n_t+1)
            psorted, sorted_pred_ids = torch.sort(preds, descending=True, stable=True)
            change_mask = torch.ones(preds.shape[0], device=device, dtype=torch.bool)
            change_mask[1:] = psorted[:-1] - psorted[1:] > 1e-3
            group_positions = torch.nonzero(change_mask, as_tuple=False).view(-1)
            idxs = torch.zeros(k, dtype=int, device=device)
            idxs[:min(k, group_positions.shape[0])] = group_positions[:min(k, group_positions.shape[0])]
            idxs = sorted_pred_ids[idxs] 
            u = instance.edge_label_index[0][idxs]
            v = instance.edge_label_index[1][idxs] - n_s
            rows = torch.arange(k, device=device, dtype=torch.long)
            uu = (u + (n_s+n_t+1) * rows).long() 
            vv = (v + (n_s+n_t+1) * rows + n_s).long() 
            unmatched_s[rows, u] = False
            unmatched_t[rows, v] = False
            matching[rows, u] = v
            xs = instance.x.repeat(k, 1)
            edge_indexes = instance.edge_index.repeat(1, k) + torch.arange(k, device=device).repeat_interleave(instance.edge_index.shape[1]) * (n_s+n_t+1)
            edge_attrs = instance.edge_attr.repeat(k, 1)
            new_edge_index = torch.zeros((2, 2*k), device=device, dtype=torch.long)
            new_edge_attr = torch.zeros((2*k, edge_attrs.shape[1]), device=device)
            new_edge_attr[:, -1] = 1
            new_edge_index[:, :k] = torch.stack([uu, vv], dim=0)
            new_edge_index[:, k:] = torch.stack([vv, uu], dim=0)
            edge_indexes = torch.cat( [edge_indexes, new_edge_index ], dim=-1 )
            edge_attrs = torch.cat([edge_attrs, new_edge_attr], dim=0)
            matching_index = torch.stack([uu, vv], dim=0)
            unmatched = torch.cat([unmatched_s, unmatched_t, torch.zeros((k,1))], dim=-1).view(-1)
            assert unmatched.shape[0] == xs.shape[0]
            instance = Data(x=xs, edge_index=edge_indexes, edge_attr=edge_attrs, edge_label_index=edge_label_index, matching_index=matching_index, unmatched=unmatched) 
            instanceL[b] = instance
        N_s = max(n_sL)
        N_t = max(n_tL)
        batchs = torch.cat(batchs).to(ddevice)
        instance = Batch.from_data_list(instanceL).to(ddevice)
        n_s_tensor = torch.tensor(n_sL, device=ddevice, dtype=torch.long)
        n_t_tensor = torch.tensor(n_tL, device=ddevice, dtype=torch.long)
        unmatched_s_tensor = torch.zeros((B, k, N_s), dtype=bool, device=ddevice)
        unmatched_t_tensor = torch.zeros((B, k, N_t), dtype=bool, device=ddevice)
        matching = torch.full((B, k, N_s), -2, device=ddevice, dtype=torch.long)
        for b in range(B):
            n_s = n_sL[b]
            n_t = n_tL[b]
            unmatched_s_tensor[b, :, :n_s] = unmatched_sL[b] 
            unmatched_t_tensor[b, :, :n_t] = unmatched_tL[b] 
            matching[b, :, :n_s] = matchingL[b]
        node_offsets = node_offsets.to(ddevice)
        for _ in range(N_s-1):
            has_unmatched_neigh = scatter_max( instance.unmatched[instance.edge_index[1]], instance.edge_index[0], dim=0, dim_size=instance.x.shape[0])[0].to(torch.bool) 
            has_unmatched_neigh = has_unmatched_neigh | instance.unmatched.to(torch.bool)  
            s_idx, t_idx = instance.matching_index
            self_and_match_have_unmatched_neigh = (has_unmatched_neigh[s_idx] & has_unmatched_neigh[t_idx])
            mask = has_unmatched_neigh 
            mask[s_idx] = self_and_match_have_unmatched_neigh 
            mask[t_idx] = self_and_match_have_unmatched_neigh 
            src, dst = instance.edge_index
            keep = mask[src] & mask[dst]  
            instance.edge_index = instance.edge_index[:, keep]
            instance.edge_attr = instance.edge_attr[keep]
            b_idx, k_idx, i_idx, j_idx = ( unmatched_s_tensor.unsqueeze(-1) & unmatched_t_tensor.unsqueeze(-2) ).nonzero(as_tuple=True)
            start_offsets = node_offsets[b_idx * k + k_idx]
            instance.edge_label_index = torch.stack([
                start_offsets + i_idx.long(),
                start_offsets + n_s_tensor[b_idx].long() + j_idx.long(),
                ], dim=0)
            preds = self.forward(instance)
            graph_of_edge = batchs[ instance.edge_label_index[0] ]
            idxs_global = (scatter_max(preds, graph_of_edge, dim=0, dim_size=B * k)[1]) 
            idxs_global[idxs_global == graph_of_edge.shape[0]] = -1 
            uu_global = instance.edge_label_index[0, idxs_global] 
            vv_global = instance.edge_label_index[1, idxs_global]
            b_idx = torch.arange(B, device=ddevice).unsqueeze(1).expand(B, k)
            rows  = torch.arange(k, device=ddevice).unsqueeze(0).expand(B, k)
            mask = (_ < (n_s_tensor-1)).unsqueeze(1).expand(B, k)   
            uu_local = (uu_global - node_offsets).view(B, k)   
            vv_local = (vv_global - node_offsets).view(B, k) 
            b_flat    = b_idx[mask]
            rows_flat = rows[mask]
            u_flat    = uu_local[mask]
            v_flat    = (vv_local - n_s_tensor.unsqueeze(1))[mask]
            matching[b_flat, rows_flat, u_flat] = v_flat
            unmatched_s_tensor[b_flat, rows_flat, u_flat] = False
            unmatched_t_tensor[b_flat, rows_flat, v_flat] = False
            uu = instance.edge_label_index[0, idxs_global[idxs_global>=0]]
            vv = instance.edge_label_index[1, idxs_global[idxs_global>=0]]
            new_edge_index = torch.zeros((2, 2*uu.shape[0]), device=ddevice, dtype=torch.long)
            new_edge_attr = torch.zeros((2*uu.shape[0], edge_attrs.shape[1]), device=ddevice)
            new_edge_attr[:, -1] = 1
            new_edge_index[:, :uu.shape[0]] = torch.stack([uu, vv], dim=0) 
            new_edge_index[:, uu.shape[0]:] = torch.stack([vv, uu], dim=0) 
            instance.edge_index = torch.cat( [instance.edge_index, new_edge_index ], dim=-1 )
            instance.edge_attr = torch.cat([instance.edge_attr, new_edge_attr], dim=0)
            instance.matching_index = torch.cat( [instance.matching_index, torch.stack([uu, vv], dim=0)], dim=-1 )
            instance.unmatched[uu] = False
            instance.unmatched[vv] = False

        for b in range(B):
            n_s = n_sL[b]
            matchingL[b] = matching[b, :, :n_s].to(device)

        bestiL = [None for _ in range(B)]        
        costs = [ ensemble_matching_cost(sources_[b], targets_[b], matchingL[b], node_cost=self.nc, edge_cost=self.ec).tolist() for b in range(B) ]
        
        for b in range(B):
            bestv = 1e10
            besti = 0
            for k_idx in range(k):
                cost = costs[b][k_idx]
                if cost < bestv:
                    bestv = cost
                    besti = k_idx
            bestiL[b] = besti
        
        return [ matchingL[b][bestiL[b]] for b in range(B) ]


