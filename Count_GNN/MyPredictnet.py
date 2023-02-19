import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import map_activation_str_to_layer, batch_convert_len_to_mask, mask_seq_by_len, extend_dimensions, gather_indices_by_lens

class PatternReadout(torch.nn.Module):
    def __init__(self,input_channels,output_channels,act_func,dropout):
        super(PatternReadout,self).__init__()
        self.W_readout_p=torch.nn.Parameter(torch.FloatTensor(input_channels, output_channels))
        self.input_c=input_channels
        self.output_c=output_channels
        self.bais_readout_p = torch.nn.Parameter(torch.FloatTensor(1,output_channels))
        self.act=act_func
        self.dropout=dropout
        self.reset_parameters()

    def reset_parameters(self):
        '''torch.nn.init.xavier_uniform_(self.W_readout_p)
        #torch.nn.init.xavier_uniform_(self.bais_readout_p)
        torch.nn.init.zeros_(self.bais_readout_p)'''
        torch.nn.init.xavier_uniform_(self.W_readout_p)
        torch.nn.init.zeros_(self.bais_readout_p)

    def forward(self,edge_attr):
        #mean
        mean = edge_attr.mean(dim=1)
        result = torch.matmul(mean, self.W_readout_p) + self.bais_readout_p
        result = F.dropout(result, p=self.dropout, training=self.training)
        result=self.act(result)
        return result


class GraphReadout(torch.nn.Module):
    def __init__(self,input_channels, output_channels,act_func):
        super(GraphReadout,self).__init__()
        self.W_gama=torch.nn.Parameter(torch.FloatTensor(input_channels, input_channels))
        self.U_gama=torch.nn.Parameter(torch.FloatTensor(input_channels, input_channels))
        self.bais_gama = torch.nn.Parameter(torch.FloatTensor(1,input_channels))
        self.W_beta=torch.nn.Parameter(torch.FloatTensor(input_channels, input_channels))
        self.U_beta=torch.nn.Parameter(torch.FloatTensor(input_channels, input_channels))
        self.bais_beta = torch.nn.Parameter(torch.FloatTensor(1,input_channels))

        self.W_readout_g=torch.nn.Parameter(torch.FloatTensor(input_channels, output_channels))
        self.bais_readout_g = torch.nn.Parameter(torch.FloatTensor(1,output_channels))

        self.input_c=input_channels
        self.output_c=output_channels
        self.act=act_func

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_gama)
        torch.nn.init.xavier_uniform_(self.U_gama)
        torch.nn.init.xavier_uniform_(self.bais_gama)
        torch.nn.init.xavier_uniform_(self.W_beta)
        torch.nn.init.xavier_uniform_(self.U_beta)
        torch.nn.init.xavier_uniform_(self.bais_beta)
        torch.nn.init.xavier_uniform_(self.W_readout_g)
        torch.nn.init.xavier_uniform_(self.bais_readout_g)

    def forward(self,edge_attr, pattern_embedding):
        #mean
        #formula(9)(10)(8)
        one=torch.ones(len(edge_attr), len(edge_attr[0]), self.input_c)

        ################################################
        ####################sum#########################
        ################################################
        '''gama = F.leaky_relu(torch.matmul(edge_attr,self.W_gama)+torch.matmul(one*pattern_embedding,self.U_gama)+one*self.bais_gama)
        beta = F.leaky_relu(torch.matmul(edge_attr,self.W_beta)+torch.matmul(one*pattern_embedding,self.U_beta)+one*self.bais_beta)'''
        gama = self.act(torch.matmul(edge_attr,self.W_gama)+one*torch.matmul(pattern_embedding,self.U_gama)+one*self.bais_gama)
        beta = self.act(torch.matmul(edge_attr,self.W_beta)+one*torch.matmul(pattern_embedding,self.U_beta)+one*self.bais_beta)
        _edge_attr = (gama+one)*edge_attr+beta
        mean = _edge_attr.mean(dim=1)
        result = torch.matmul(mean, self.W_readout_g) + self.bais_readout_g
        result=self.act(result)
        return  result,gama,beta


class Counter_13(torch.nn.Module):
    def __init__(self,length,act_fun,dropout):
        super(Counter_13,self).__init__()
        self.W_c_13=torch.nn.Parameter(torch.FloatTensor(length,length))
        self.bias_c_13=torch.nn.Parameter(torch.FloatTensor(1,length))
        self.reset_parameters()
        self.length=length
        self.act=act_fun
        self.dropout=dropout
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_c_13)
        torch.nn.init.xavier_uniform_(self.bias_c_13)

    def forward(self,pattern_embedding,graph_embedding):
        p_e=pattern_embedding.reshape(self.length,1)
        g_e=graph_embedding.reshape(1,self.length)
        Relu=F.relu(torch.matmul(g_e,self.W_c_13)+self.bias_c_13)
        #result=F.leaky_relu(torch.matmul(leakyR,p_e))
        result = torch.matmul(Relu, p_e)
        result=torch.squeeze(result)
        result=F.dropout(result, p=self.dropout, training=self.training)
        result=self.act(result)
        return result
