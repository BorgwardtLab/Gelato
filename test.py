import os
import argparse
import numpy as np
import random
import statistics
import time
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.dataset import GraphMatchingDataset
from src.model import LinkGNN
from src.utils import run_inference
from src.utils import normalized_mae, exact_hit_rate




def main(args):
    random.seed(0)

    dataset_inf = GraphMatchingDataset(name=args.data, root=args.root, num_pairs=args.num_samples, split=args.split, bounds=args.size_bounds)
    
    num_node_labels = dataset_inf[0].x_s.shape[1]+2
    num_edge_labels = dataset_inf[0].edge_attr_s.shape[1]+1

    model = LinkGNN(num_node_labels, num_edge_labels, 128, args.layers, args.node_cost, args.edge_cost)

    model.load_state_dict(torch.load(args.load_ckp))
    model = model.to(args.device)
    model.eval()

    maeL, nmaeL, rmseL, ehrL, rtimeL = [], [], [], [], []
    for seed in range(5):   
        dataset_inf = GraphMatchingDataset(name=args.data, root=args.root, num_pairs=args.num_samples, split=args.split, bounds=args.size_bounds, seed=seed)
        startt = time.time()
        costs, true_costs = run_inference(model, dataset_inf, args.k, batch_size=32)

        rtime = 1000.0 * (time.time() - startt) / len(dataset_inf)
        mae = mean_absolute_error(costs, true_costs)
        nmae = normalized_mae(costs, true_costs) * 100.0
        rmse = root_mean_squared_error(costs, true_costs)
        ehr = exact_hit_rate(costs, true_costs) * 100.0
        maeL.append(mae)
        nmaeL.append(nmae)
        rmseL.append(rmse)
        ehrL.append(ehr)
        rtimeL.append(rtime)
        print(f"MAE {mae:.3f}, nMAE {nmae:.3f}, RMSE {rmse:.3f}, EHR {ehr:.3f} Time {rtime:.3f}")
        if args.log:
            with open(args.log, "a") as f:
                f.write(f"MAE {mae:.3f}, nMAE {nmae:.3f}, RMSE {rmse:.3f}, EHR {ehr:.3f}\n")
    
    mae_m, mae_s = statistics.mean(maeL), statistics.stdev(maeL)
    nmae_m, nmae_s = statistics.mean(nmaeL), statistics.stdev(nmaeL)
    rmse_m, rmse_s = statistics.mean(rmseL), statistics.stdev(rmseL)
    ehr_m, ehr_s = statistics.mean(ehrL), statistics.stdev(ehrL)
    rtime_m, rtime_s = statistics.mean(rtimeL), statistics.stdev(rtimeL)
    
    print(f"nMAE {nmae_m:.1f}±{nmae_s:.1f}, EHR {ehr_m:.1f}±{ehr_s:.1f},  MAE {mae_m:.3f}±{mae_s:.3f} , RMSE {rmse_m:.3f}±{rmse_s:.3f},  Time {rtime_m:.1f}±{rtime_s:.1f}")
    if args.log:
        with open(args.log, "a") as f:
            f.write(f"MAE {mae_m:.3f} {mae_s:.3f}, nMAE {nmae_m:.3f} {nmae_s:.3f}, RMSE {rmse_m:.3f} {rmse_s:.3f}, EHR {ehr_m:.3f} {ehr_s:.3f},  Time {rtime_m:.3f} {rtime_s:.3f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--root', type=str, default='data/')
    parser.add_argument('--data', type=str, default=None)
    
    parser.add_argument('--size_bounds', type=int, nargs=2, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'larger'])
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=32)

    parser.add_argument('--node_cost', type=float, default=1.0)
    parser.add_argument('--edge_cost', type=float, default=1.0)

    parser.add_argument('--load_ckp', type=str, default=None)
    parser.add_argument('--nocuda', action='store_true')
    parser.add_argument('--log', type=str, default=None)

    args = parser.parse_args()

    args.device = torch.device("cuda" if (torch.cuda.is_available() and (not args.nocuda)) else "cpu")

    print(args.device)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(linewidth=200)
    torch.set_printoptions(edgeitems=20)

    if args.log:
        args.log = os.path.join( args.log, args.load_ckp.split('/')[-1].rsplit('.', 1)[0]+"_"+str(args.size_bounds)+"_k"+str(args.k)+"_inf.log" )
        with open(args.log, "w") as f:
            f.write(str(args)+'\n')

    main(args)