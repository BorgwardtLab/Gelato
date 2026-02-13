# Gelato

Datasets and code for the paper: "Gelato: Graph Edit Distance via Autoregressive Neural Combinatorial Optimization", published at ICLR 2026. The paper is available [here](https://openreview.net/forum?id=6ZTcLNmguc).

The repository provides code for training and testing the Gelato model. 

Moreover, ```src/dataset.py``` contains a dataset class with the GED datasets used in the paper. We provide pre-computed train-val-test splits with no data leakage, ground-truth optimal matchings, and out-of-distribution data in the ```larger``` data split. 

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

### Citing our work

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
