import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import copy
from functools import partial
from dgl.nn.pytorch.conv import RelGraphConv
from basemodel import GraphAdjModel
from utils import map_activation_str_to_layer, split_and_batchify_graph_feats, GetEdgeAdj, split_batch
from MyModel import MeanN


class EdgeMean(GraphAdjModel):
    def __init__(self, config):
        super(EdgeMean, self).__init__(config)

        #self.ignore_norm = config["rgcn_ignore_norm"]

        # create networks

        p_emb_dim, g_emb_dim, p_e_emb_dim, g_e_emb_dim = self.get_emb_dim()

        self.g_net, g_dim = self.create_net(
            name="graph", input_dim=g_emb_dim*3, hidden_dim=config["ppn_hidden_dim"],
            num_layers=config["ppn_graph_num_layers"],  act_func=self.act_func,
            dropout=self.dropout, bsz=config["batch_size"])

        self.p_net, p_dim = (self.g_net, g_dim) if self.share_arch else self.create_net(
            name="pattern", input_dim=p_emb_dim*3, hidden_dim=config["ppn_hidden_dim"],
            num_layers=config["ppn_pattern_num_layers"], act_func=self.act_func,
            dropout=self.dropout, bsz=config["batch_size"])

        # create predict layers

        if self.add_enc:

            p_enc_dim, g_enc_dim,p_e_enc_dim,g_e_enc_dim = self.get_enc_dim()
            p_dim += p_enc_dim*2+p_e_enc_dim
            g_dim += g_enc_dim*2+g_e_enc_dim
        if self.add_degree:
            p_dim += 2
            g_dim += 2
        self.predict_net = self.create_predict_net(config["predict_net"],
            pattern_dim=p_dim, graph_dim=g_dim, hidden_dim=config["predict_net_hidden_dim"],
            num_heads=config["predict_net_num_heads"], recurrent_steps=config["predict_net_recurrent_steps"],
            mem_len=config["predict_net_mem_len"], mem_init=config["predict_net_mem_init"])

        self.g_linear=torch.nn.Linear(g_emb_dim*3 ,config["ppn_hidden_dim"])
        self.p_linear=torch.nn.Linear(p_emb_dim*3 ,config["ppn_hidden_dim"])
        self.config=config
    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        hidden_dim = kw.get("hidden_dim", 64)
        act_func = kw.get("act_func", "relu")
        dropout = kw.get("dropout", 0.0)
        bsz = kw.get("bsz",112)
        ppns = nn.ModuleList()
        for i in range(num_layers):
            ppns.add_module("%s_rgc%d" % (name, i), MeanN(
                input_channels=hidden_dim if i > 0 else input_dim, output_channels=hidden_dim,
                act_func=map_activation_str_to_layer(act_func), dropout=dropout, bsz=bsz))
        return ppns, hidden_dim

    def GraphEmbedding(self,g_vl_emb,g_el_emb,adj):
        u=g_vl_emb[adj[0]]
        v=g_vl_emb[adj[1]]
        result=torch.cat([u,g_el_emb,v],dim=1)
        return result

    def PredictEnc(self,edge_enc,pattern_enc, adj):
        u=pattern_enc[adj[0]]
        v=pattern_enc[adj[1]]
        result=torch.cat([u,edge_enc,v],dim=1)
        return result

    def CatIndeg(self,indeg,adj):
        u=indeg[adj[0]]
        v=indeg[adj[1]]
        result=torch.cat([u,v],dim=1)
        return result

    def forward(self, pattern, pattern_len, pattern_e_len, graph, graph_len, graph_e_len, p_e_max, g_e_max,p_index,g_index):
        bsz = pattern_len.size(0)
        zero_mask=None

        p_vl_emb, g_vl_emb, p_el_emb,g_el_emb = self.get_emb(pattern, graph)
        pattern_output = self.GraphEmbedding(p_vl_emb,p_el_emb,pattern.adjacency_matrix()._indices())
        pattern_eadj=pattern.edata["eadj"]
        pattern_input=split_batch(p_index,pattern_output,pattern_e_len,p_e_max)
        pattern_eadj=split_batch(p_index,pattern_eadj,pattern_e_len,p_e_max)
        pattern_first=self.p_linear(pattern_input)

        for p_rgcn in self.p_net:
            o = p_rgcn(pattern_input, pattern_eadj, p_e_max)
            pattern_output = o + pattern_first
            pattern_first=pattern_output

        graph_output = self.GraphEmbedding(g_vl_emb,g_el_emb,graph.adjacency_matrix()._indices())
        if zero_mask is not None:
            graph_output.masked_fill_(zero_mask, 0.0)


        graph_eadj=graph.edata["eadj"]
        graph_input=split_batch(g_index,graph_output,graph_e_len,g_e_max)
        graph_eadj=split_batch(g_index,graph_eadj,graph_e_len,g_e_max)
        graph_first = self.g_linear(graph_input)
        zero_output_mask=None
        for g_rgcn in self.g_net:
            o = g_rgcn(graph_input, graph_eadj, g_e_max, zero_output_mask)
            graph_output = o + graph_first
            graph_first=graph_output
            #graph_output.masked_fill_(zero_output_mask, 0.0)
        #graph_output = graph_output.resize(graph_output.size(0) * graph_output.size(1), graph_output.size(2))

        '''if zero_mask is not None:
            graph_output.masked_fill_(zero_mask, 0.0)'''
 
        if self.add_enc and self.add_degree:

            pattern_enc, graph_enc,pattern_e_enc,graph_e_enc = self.get_enc(pattern, graph)
            p_enc = self.PredictEnc(pattern_e_enc,pattern_enc,pattern.adjacency_matrix()._indices())
            g_enc = self.PredictEnc(graph_e_enc,graph_enc,graph.adjacency_matrix()._indices())
            p_indeg=self.CatIndeg(pattern.ndata["indeg"].unsqueeze(-1),pattern.adjacency_matrix()._indices())
            g_indeg=self.CatIndeg(graph.ndata["indeg"].unsqueeze(-1),graph.adjacency_matrix()._indices())
            p_enc=split_batch(p_index,p_enc,pattern_e_len,p_e_max).reshape(-1,p_enc.size(1))
            g_enc=split_batch(g_index,g_enc,graph_e_len,g_e_max).reshape(-1,g_enc.size(1))
            p_indeg=split_batch(p_index,p_indeg,pattern_e_len,p_e_max).reshape(-1,p_indeg.size(1))
            g_indeg=split_batch(g_index,g_indeg,graph_e_len,g_e_max).reshape(-1,g_indeg.size(1))
            pattern_output=pattern_output.reshape(-1,pattern_output.size(2))
            graph_output=graph_output.reshape(-1,graph_output.size(2))
            pattern_output=torch.cat([p_enc,pattern_output,p_indeg],dim=1)
            graph_output=torch.cat([g_enc,graph_output,g_indeg],dim=1)
            pattern_output=pattern_output.reshape(bsz,-1,pattern_output.size(1))
            graph_output=graph_output.reshape(bsz,-1,graph_output.size(1))

        elif self.add_enc:
            pattern_enc, graph_enc,pattern_e_enc,graph_e_enc = self.get_enc(pattern, graph)
            if zero_mask is not None:
                graph_enc.masked_fill_(zero_mask, 0.0)
            pattern_output = torch.cat([pattern_enc, pattern_output], dim=1)
            graph_output = torch.cat([graph_enc, graph_output], dim=1)
        elif self.add_degree:
            pattern_output = torch.cat([pattern_output, pattern.ndata["indeg"].unsqueeze(-1)], dim=1)
            graph_output = torch.cat([graph_output, graph.ndata["indeg"].unsqueeze(-1)], dim=1)

        #pred = self.predict_net(pattern_output, pattern_e_len,graph_output, graph_e_len)
        #return pred
        pred,alpha,beta = self.predict_net(pattern_output, pattern_e_len,graph_output, graph_e_len)
        filmreg=(torch.sum(alpha**2))**0.5+(torch.sum(beta**2))**0.5
        return pred,filmreg
