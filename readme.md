# Gelato

Datasets and code for the paper: "Gelato: Graph Edit Distance via Autoregressive Neural Combinatorial Optimization", published at ICLR 2026. The paper is available [here](https://openreview.net/forum?id=6ZTcLNmguc).

The repository provides code for training and testing the Gelato model. 
Moreover, ```src/dataset.py``` contains a dataset class with the GED datasets used in the paper. 

### Training

The following commands can be used to train Gelato on the main datasets used in the paper.
```
python train.py --data aids --save_ckp checkpoints/model_aids.pt --train_pairs 88000
python train.py --data linux --save_ckp checkpoints/model_linux.pt --train_pairs 25000
python train.py --data imdb-16 --save_ckp checkpoints/model_imdb.pt --train_pairs 25000
python train.py --data zinc-16 --save_ckp checkpoints/model_zinc.pt --train_pairs 125000
python train.py --data molhiv-16 --save_ckp ckp/model_molhiv.pt --train_pairs 200000
python train.py --data code2-22 --save_ckp ckp/model_code.pt --train_pairs 100000
```

### Testing

Checkpoints for Gelato are available in the ```checkpoints/``` folder. 

Example usage for in-distribution testing: 
```
python test.py --data zinc-16 --load_ckp checkpoints/model_zinc.pt
```
Example usage for out-of-distribution testing:
```
python test.py --data zinc-16 --load_ckp checkpoints/model_zinc.pt --split larger --size_bounds 17 18 --num_samples 500
```

## Datasets

In ```src/dataset.py```, we provide easy-to-use datasets with several improvements over existing ones:
- We provide pre-computed train-val-test splits with **no data leakage** (due to graph isomorphism) across splits.
- The datasets have both edge-labeled and edge-unlabeled variants of graphs.
- We provide optimal solutions for graphs **up to 30 nodes** to test for out-of-distribution generalization, in the ```larger``` data split.

Example usage:

```python
from src.dataset import GraphMatchingDataset

# Get 1000 graph pairs from the test split of the AIDS dataset
dataset = GraphMatchingDataset(name='aids', root='data/', num_pairs=1000, split='test')

# Get 1000 graph pairs from the 'larger' split of the ZINC-16 dataset
dataset = GraphMatchingDataset(name='zinc-16', root='data/', num_pairs=1000, split='larger')

# Get 1000 graph pairs with graphs between 23 and 26 nodes from the 'larger' split of the code2-22 dataset
dataset = GraphMatchingDataset(name='code2-22', root='data/', num_pairs=1000, split='larger', bounds=(23, 26))

for data in dataset:
  graph_1 = Data(x=data.x_s, edge_index=data.edge_index_s, edge_attr=data.edge_attr_s)
  graph_2 = Data(x=data.x_t, edge_index=data.edge_index_t, edge_attr=data.edge_attr_t)
  optimal_matching = data.matching.long()
```


## Citing our work

Please cite our ICLR 2026 paper in case you find Gelato useful for your applications.

> Paolo Pellizzoni, Till Hendrik Schulz, and Karsten Borgwardt. _Gelato: Graph Edit Distance via Autoregressive Neural Combinatorial Optimization_, in ICLR, 2026.

```
@inproceedings{
  pellizzoni2026gelato,
  title={Gelato: Graph Edit Distance via Autoregressive Neural Combinatorial Optimization},
  author={Paolo Pellizzoni and Till Hendrik Schulz and Karsten Borgwardt},
  booktitle={International Conference on Learning Representations},
  year={2026},
}
```
