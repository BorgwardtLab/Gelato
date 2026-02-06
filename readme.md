# Gelato

Datasets and code for the paper: "Gelato: Graph Edit Distance via Autoregressive Neural Combinatorial Optimization", published at ICLR 2026. The paper is available [here](https://openreview.net/forum?id=6ZTcLNmguc).

The repository provides code for training and testing the Gelato model. 

Moreover, ```src/dataset.py``` contains a dataset class with the GED datasets used in the paper. We provide pre-computed train-val-test splits with no data leakage, ground-truth optimal matchings, and out-of-distribution data in the ```larger``` data split. 

### Training

Example usage: 
```
python test.py --data data/aids/ --load_ckp checkpoints/model_aids.pt --k 32
```

### Testing

Checkpoints for Gelato are available in the ```checkpoints/``` folder. 

Example usage for in-distribution testing: 
```
python test.py --data data/zinc-16/ --load_ckp checkpoints/model_zinc.pt --k 32
```
Example usage for out-of-distribution testing:
```
python test.py --data data/zinc-16/ --load_ckp checkpoints/model_zinc.pt --k 32 --split larger --size_bounds 17 18 --num_samples 500
```

### Citing our work

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
