import argparse
import numpy as np
import random
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.dataset import GraphMatchingDataset
from src.subproblem_dataset import GraphMatchingSubproblemDataset
from src.model import LinkGNN
from src.utils import run_inference, training_step_link, validation_step_link
from src.utils import normalized_mae, exact_hit_rate



def main(args):
    
    train_dataset = GraphMatchingSubproblemDataset(name=args.data, num_pairs=args.train_pairs, num_instances_per_pair=args.instances_per_pair, split='train')
    val_dataset = GraphMatchingSubproblemDataset(name=args.data, num_pairs=100, num_instances_per_pair=args.instances_per_pair, split='val')
    val_dataset_inf = GraphMatchingDataset(name=args.data, num_pairs=2000, split='val')
    
    num_node_labels = train_dataset[0].x.shape[1]
    num_edge_labels = train_dataset[0].edge_attr.shape[1]
 
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, pin_memory=True)
    

    model = LinkGNN(num_node_labels, num_edge_labels, 128, args.layers, args.node_cost, args.edge_cost)
    model = model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_nmae = 1e10
    for epoch in tqdm(range(args.epochs), ncols=64):
        epoch_loss, epoch_acc = training_step_link(model, train_loader, optimizer, args)
        epoch_val_loss, epoch_val_acc = validation_step_link(model, val_loader, args)
        
        costs, true_costs = run_inference(model, val_dataset_inf, k=args.k, batch_size=64, disable_tqdm=True)

        nmae = normalized_mae(costs, true_costs)
        ehr = exact_hit_rate(costs, true_costs)
        print(f'train-loss: {epoch_loss:.5f}, nMAE: {nmae:.5f}, EHR: {ehr:.5f}')
        if args.log and args.save_ckp:
            with open(args.save_ckp.rsplit('.', 1)[0]+"_train.log", "a") as f:
                f.write(f'{epoch_loss:.6f} {epoch_acc:.5f}  {epoch_val_loss:.6f} {epoch_val_acc:.5f}  {nmae:.5f} {ehr:.5f}\n')
        if args.save_ckp is not None and (nmae < best_nmae):
            best_nmae = nmae
            torch.save(model.to('cpu').state_dict(), args.save_ckp)
            model.to(args.device)
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_train_steps', type=float, default=1.0)

    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--train_pairs', type=int, default=None)
    parser.add_argument('--instances_per_pair', type=int, default=40)
    parser.add_argument('--node_cost', type=float, default=1.0)
    parser.add_argument('--edge_cost', type=float, default=1.0)

    parser.add_argument('--save_ckp', type=str, default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--nocuda', action='store_true')

    args = parser.parse_args()


    args.device = torch.device("cuda" if (torch.cuda.is_available() and (not args.nocuda)) else "cpu")
    print(args)
    if args.log and args.save_ckp:
        with open(args.save_ckp.rsplit('.', 1)[0]+"_train.log", "w") as f:
            f.write(str(args)+'\n')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(linewidth=200)
    torch.set_printoptions(edgeitems=20)

    main(args)