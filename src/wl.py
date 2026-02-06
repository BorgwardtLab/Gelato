import torch
from torch_scatter import scatter_add

def mash(input, modulo=1000000007):
    output = torch.sum(input*torch.tensor([2 ** ((input.shape[1]-1-i)%57) for i in range(input.shape[1])]), dim=1, dtype=torch.int64) 
    return output % modulo

def mix(x: torch.Tensor) -> torch.Tensor:
    return ((x >> 16) ^ x) * torch.tensor(0x119de1f3, dtype=torch.int64, device=x.device)

class WeisfeilerLeman:
    """
    Implements the classical 1-dimensional Weisfeiler–Leman (WL) color refinement algorithm
    that iteratively updates node labels based on multiset of neighbor labels and edge attributes.
    """
    def __init__(self, num_iterations: int = 3):
        self.num_iterations = num_iterations

    def _initialize_labels(self, x: torch.Tensor):
        # Hash initial node features to integer labels starting from 0
        device = x.device
        feat_tuples = [tuple(fe.tolist()) for fe in x]
        label_map = {}
        labels = []
        next_id = 0
        for t in feat_tuples:
            if t not in label_map:
                label_map[t] = next_id
                next_id += 1
            labels.append(label_map[t])
        return torch.tensor(labels, dtype=torch.long, device=device)

    def _initialize_edge_labels(self, edge_attr: torch.Tensor):
        """
        Hash edge feature vectors to integer labels starting from 0.
        Returns a tensor of shape [M] dtype=torch.long on the same device as edge_attr.
        """
        device = edge_attr.device
        feat_tuples = [tuple(fe.tolist()) for fe in edge_attr]
        label_map = {}
        labels = []
        next_id = 0
        for t in feat_tuples:
            if t not in label_map:
                label_map[t] = next_id
                next_id += 1
            labels.append(label_map[t])
        return torch.tensor(labels, dtype=torch.long, device=device)

    def run(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        N = x.size(0)
        labels = mash(x)
        edge_labels = mash(edge_attr)

        src, dst = edge_index
        cnt = 0
        for _ in range(self.num_iterations):
            edge_values = mix( labels[src] + mix(edge_labels) ) 
            signatures = scatter_add(edge_values, dst, dim=0, dim_size=N)
            labels = mix(signatures) + labels

        return labels

    
    def run_exact(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        x:        Tensor of shape [N, d]    (node features)
        edge_index: LongTensor of shape [2, M]
        edge_attr:  Tensor of shape [M, d2] (edge features)

        Returns:
            labels: Tensor of shape [N] with final WL labels (integers starting at 0 per iteration)
        """
        N = x.size(0)
        # Initialize labels from node features
        labels = self._initialize_labels(x)
        # Initialize integer labels for edge attributes
        edge_labels = self._initialize_edge_labels(edge_attr)
        
        
        src, dst = edge_index
        cnt = 0
        for _ in range(self.num_iterations):
            # Map signatures to new labels for this iteration
            local_map = {}
            next_id = 0
            new_labels = torch.empty_like(labels)

            # Gather neighbor signatures (use integer edge labels now)
            signatures = [[] for _ in range(N)]
            for e in range(edge_index.size(1)):
                u = int(src[e]); v = int(dst[e])
                signatures[v].append( 10000 * int(labels[u]) + int(edge_labels[e]) )

            # Assign new labels based on sorted signature + current label
            for v in range(N):
                sig = tuple(sorted(signatures[v]))
                full_sig = (int(labels[v]), sig)
                if full_sig not in local_map:
                    local_map[full_sig] = next_id
                    next_id += 1
                new_labels[v] = local_map[full_sig]

            labels = new_labels

            if len(local_map) == cnt:
                break 
            cnt = len(local_map)

        return labels
