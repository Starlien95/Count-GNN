import os
import tqdm
import numpy
from utils import GetEdgeAdj
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader


def edge_adj_load(pattern,graph):
    pattern_edgeadj=dict()
    graph_edgeadj=dict()
    data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
    for data_type in data_loaders:
        pattern_edgeadj[data_type] = list()
        graph_edgeadj[data_type] = list()
    for batch in tqdm.trange(112):
        for j in range(8):
            loaded_data = numpy.load(os.path.join("./data/debug/edgeadj/train", "%s_edgeadj.npz" % (pattern["train"][batch * 8 + j].id)))  # 读取含有多个数组的文件
            pattern_edgeadj["train"].append(torch.from_numpy(loaded_data['arr_0']))
            graph_edgeadj["train"].append(torch.from_numpy(loaded_data['arr_1']))

    for batch in tqdm.trange(112):
        loaded_data = numpy.load(os.path.join("./data/debug/edgeadj/test",
                                              "%s_edgeadj.npz" % (pattern["test"][batch].id)))  # 读取含有多个数组的文件
        pattern_edgeadj["test"].append(torch.from_numpy(loaded_data['arr_0']))
        graph_edgeadj["test"].append(torch.from_numpy(loaded_data['arr_1']))

    for batch in tqdm.trange(112):
        loaded_data = numpy.load(os.path.join("./data/debug/edgeadj/dev",
                                              "%s_edgeadj.npz" % (pattern["dev"][batch].id)))  # 读取含有多个数组的文件
        pattern_edgeadj["dev"].append(torch.from_numpy(loaded_data['arr_0']))
        graph_edgeadj["dev"].append(torch.from_numpy(loaded_data['arr_1']))

    dataloader_peadj= DataLoader(pattern_edgeadj)
    dataloader_geadj=DataLoader(graph_edgeadj)

    return dataloader_peadj,dataloader_geadj

'''dataload_p,dataload_g=pygdataload()

dataload_peadj,dataload_geadj=edge_adj_load(dataload_p.dataset,dataload_g.dataset)
print(dataload_peadj.dataset["train"][0])'''

