
# Count-GNN
We provide the code (in pytorch) and datasets for our paper [**"Learning to Count Isomorphisms with Graph Neural Networks"**](https://arxiv.org/pdf/2302.03266.pdf), which is accepted by AAAI23.

## Description
The repository is organised as follows:
- **datasets/data/**: contains data we use. Need to be decompressed and be placed in the same path as Count_GNN/
- **Count_GNN/**: contains our model.
- **converter/**: transform the original dataset into the data format that can be inputted into Count_GNN.
- **generator/**: generate synthetic dataset.
- **Appendix.pdf**: Appendix of our paper [**"Learning to Count Isomorphisms with Graph Neural Networks"**](https://arxiv.org/pdf/2302.03266.pdf)

## Package Dependencies

* tqdm
* numpy
* pandas
* scipy
* tensorboardX
* torch >= 1.3.0
* dgl == 0.4.3post2

## Running experiments

* **train model**:
python _train.py --model EDGEMEAN --predict_net FilmSumPredictNet --emb_dim 4 --ppn_hidden_dim 12 --weight_decay_film 0.0001
* **test model**:
python evaluate.py ../dumps/MUTAG

## Citation
@inproceedings{yu2023learning,\
  title={Learning to Count Isomorphisms with Graph Neural Networks},\
  author={Yu, Xingtong and Liu, Zemin and Fang, Yuan and Zhang, Xinming},\
  booktitle={Proceedings of the AAAI conference on artificial intelligence},\
  year={2023}\
}
