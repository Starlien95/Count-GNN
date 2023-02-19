
# Count-GNN
We provide the implementaion of Count-GNN model.

The repository is organised as follows:
- data/: contains datasets.
- Count_GNN/: contains our model.
- converter/: transform the original dataset into the data format that can be inputted into Count_GNN.
- generator/: generate synthetic dataset.

## Package Dependencies

* tqdm
* numpy
* pandas
* scipy
* tensorboardX
* torch >= 1.3.0
* dgl == 0.4.3post2

## Running experiments

To run _train.py:
python _train.py --model EDGEMEAN --predict_net FilmSumPredictNet --emb_dim 4 --ppn_hidden_dim 12 --weight_decay_film 0.0001
To run evaluate.py:
python evaluate.py ../dumps/MUTAG

## Citation
* Title： Learning to Count Isomorphisms with Graph Neural Networks
* Author: Xingtong Yu*, Zemin Liu*, Yuan Fang, Xinming Zhang
* In proceedings: AAAI23
