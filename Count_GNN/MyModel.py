import torch
import dgl
import torch.nn.functional as F
import numpy as np
from utils import map_activation_str_to_layer

class GATNeigh_Agg(torch.nn.Module):
    def __init__(self,in_channels,act_func,dropuout):
        super(GATNeigh_Agg, self).__init__()
        self.W_2 = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self.U_2 = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.yita = torch.nn.Parameter(torch.Tensor(in_channels,1))
        self.act = act_func
        self.droput=dropuout

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_2)
        torch.nn.init.xavier_uniform_(self.U_2)
        torch.nn.init.xavier_uniform_(self.yita)

    def forward(self,edge_attr, edge_adj, e_max, mask):
        #alpha[edge_num,edge_num]
#        print('in_channels:',self.in_channels)
 #       print('edge_attr:',edge_attr.size())
  #      print('U:',self.U_2.size())
        v_u_matrix=torch.matmul(edge_attr,self.U_2)
        i_v_matrix=torch.matmul(edge_attr,self.W_2)

        #print(mask.device)
        #e_max=torch.tensor(e_max,device=torch.device("cuda:1"))
        bsz=v_u_matrix.size(0)
        edge_num=edge_attr.size(1)
        v_u_matrix=v_u_matrix.expand(edge_num, bsz, edge_num,self.out_channels)
        v_u_matrix=v_u_matrix.permute(1,2,0,3)
        i_v_matrix=i_v_matrix.expand(edge_num, bsz, edge_num,self.out_channels)
        i_v_matrix=i_v_matrix.permute(1,0,2,3)
        LeakyRe=self.act(v_u_matrix+i_v_matrix)
        _alpha_ivu=torch.matmul(LeakyRe,self.yita)
        _alpha_ivu=_alpha_ivu.squeeze()
        zero_vec = -1e12 * torch.ones_like(_alpha_ivu)  
        _alpha = torch.where(edge_adj > 0, _alpha_ivu, zero_vec)
        #_edge_adj=(edge_adj-1)*1e12
        #_alpha=torch.matmul(_alpha_ivu,edge_adj)+_edge_adj
        _alpha = F.softmax(_alpha, dim=2)
        _alpha=F.dropout(_alpha, p=self.droput, training=self.training)
        out=torch.matmul(_alpha,edge_attr)
        out=self.act(out)
        return out

class EdgeEmbedding_F3_4(torch.nn.Module):
    def __init__(self,input_channels,output_channels,act_func, bsz):
        super(EdgeEmbedding_F3_4, self).__init__()
        self.input_c=input_channels
        self.output_c=output_channels
        self.act = act_func
        self.bsz=bsz
        self.linear1=torch.nn.Linear(input_channels,output_channels)
        self.linear2=torch.nn.Linear(input_channels,output_channels)
        
    def forward(self,edge_attr,edge_neigh_agg):
        result=self.linear1(edge_neigh_agg)+self.linear2(edge_attr)
        result=self.act(result)
        return result


class EdgeEmbedding_15(torch.nn.Module):
    def __init__(self, input_channels, output_channels, act_func, bsz):
        super(EdgeEmbedding_15, self).__init__()
        self.input_c = input_channels
        self.output_c = output_channels
        self.act = act_func
        self.bsz = bsz
        self.beta=torch.nn.Parameter(torch.Tensor(1,1))
        self.linear = torch.nn.Linear(input_channels, output_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.beta)

    def forward(self, edge_attr, edge_neigh_agg):
        result = self.linear(edge_neigh_agg+(1+self.beta)*edge_attr)
        result = self.act(result)
        return result


class EGAT(torch.nn.Module):
    def __init__(self,input_channels,output_channels, act_func, dropout, bsz):
        super(EGAT, self).__init__()
        self.gat=GATNeigh_Agg(input_channels, act_func, dropout)
        self.emb=EdgeEmbedding_F3_4(input_channels,output_channels, act_func, bsz)
    def forward(self,edge_attr, edge_adj, e_max, mask=None):
        attr_agg=self.gat(edge_attr,edge_adj, e_max, mask)
        if mask is not None:
            attr_agg.masked_fill_(mask, 0.0)
        emb = self.emb(edge_attr,attr_agg)
        #emb = emb.resize(emb.size(0)*emb.size(1),emb.size(2))
        return emb

class EATTS(torch.nn.Module):
    def __init__(self,input_channels,output_channels, act_func, dropout, bsz):
        super(EATTS, self).__init__()
        self.gat=GATNeigh_Agg(input_channels, act_func, dropout)
        self.emb=EdgeEmbedding_15(input_channels,output_channels, act_func, bsz)
    def forward(self,edge_attr, edge_adj, e_max, mask=None):
        attr_agg=self.gat(edge_attr,edge_adj, e_max, mask)
        emb = self.emb(edge_attr,attr_agg)
        #emb = emb.resize(emb.size(0)*emb.size(1),emb.size(2))
        return emb


class Sum_Agg(torch.nn.Module):
    def __init__(self,in_channels,act_func,dropuout):
        super(Sum_Agg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.act = act_func
        self.droput=dropuout


    def forward(self,edge_attr, edge_adj, e_max, mask):
        out=torch.matmul(edge_adj,edge_attr)
        out=F.dropout(out, p=self.droput, training=self.training)
        return out


class ESum(torch.nn.Module):
    def __init__(self,input_channels,output_channels, act_func, dropout, bsz):
        super(ESum, self).__init__()
        self.gat=Sum_Agg(input_channels, act_func, dropout)
        self.emb=EdgeEmbedding_F3_4(input_channels,output_channels, act_func, bsz)
    def forward(self,edge_attr, edge_adj, e_max, mask=None):
        attr_agg=self.gat(edge_attr,edge_adj, e_max, mask)
        if mask is not None:
            attr_agg.masked_fill_(mask, 0.0)
        emb = self.emb(edge_attr,attr_agg)
        #emb = emb.resize(emb.size(0)*emb.size(1),emb.size(2))
        return emb

class ESumS(torch.nn.Module):
    def __init__(self,input_channels,output_channels, act_func, dropout, bsz):
        super(ESumS, self).__init__()
        self.gat=Sum_Agg(input_channels, act_func, dropout)
        self.emb=EdgeEmbedding_15(input_channels,output_channels, act_func, bsz)
    def forward(self,edge_attr, edge_adj, e_max, mask=None):
        attr_agg=self.gat(edge_attr,edge_adj, e_max, mask)
        emb = self.emb(edge_attr,attr_agg)
        #emb = emb.resize(emb.size(0)*emb.size(1),emb.size(2))
        return emb

class Mean_Agg(torch.nn.Module):
    def __init__(self,in_channels,act_func,dropuout):
        super(Mean_Agg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.act = act_func
        self.droput=dropuout


    def forward(self,edge_attr, edge_adj, e_max, mask):
        out=torch.matmul(edge_adj,edge_attr)
        zero=torch.count_nonzero(edge_adj,dim=1)
        zero=zero.unsqueeze(2)+1
        out=out/zero
        out=F.dropout(out, p=self.droput, training=self.training)
        return out


class MeanN(torch.nn.Module):
    def __init__(self,input_channels,output_channels, act_func, dropout, bsz):
        super(MeanN, self).__init__()
        self.gat=Mean_Agg(input_channels, act_func, dropout)
        self.emb=EdgeEmbedding_F3_4(input_channels,output_channels, act_func, bsz)
    def forward(self,edge_attr, edge_adj, e_max, mask=None):
        attr_agg=self.gat(edge_attr,edge_adj, e_max, mask)
        if mask is not None:
            attr_agg.masked_fill_(mask, 0.0)
        emb = self.emb(edge_attr,attr_agg)
        #emb = emb.resize(emb.size(0)*emb.size(1),emb.size(2))
        return emb
