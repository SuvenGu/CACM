
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
    

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1) 
        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None,add=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        if add:
            q += residual 

        q = self.layer_norm(q)

        return q, attn


class Conv1DNet(nn.Module):
    def __init__(self, input_dim, out_dim,T=10,hidden = [64,128,256]):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden[0], kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden[0], out_channels=hidden[1], kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=hidden[1], out_channels=hidden[2], kernel_size=3)
        self.fc1 = nn.Linear(hidden[2]*(T-6), hidden[1])
        self.fc2 = nn.Linear(hidden[1], out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layer

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



class ClimateEmbedding(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(ClimateEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=10, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

class CropANet(nn.Module):

    def __init__(self,cfg,**kwargs):
        super(CropANet, self).__init__()
        self.in_chans = cfg["MODEL"]["IN_CHANNEL"]
        self.c_dim = cfg["MODEL"]["C_DIM"]
        self.T = cfg["MODEL"]["T"]
        self.T_a = cfg["MODEL"]["T_A"]
        self.use_cond = cfg["COND_FLAG"]
        self.num_class = cfg["MODEL"]["NUM_CLASSES"]
 
        self.kernel_t = ClimateEmbedding(2,out_dim=16)
        self.kernel_r = ClimateEmbedding(1,out_dim=16)

        self.classifier = nn.Linear(128,self.num_class)
        self.cross_attn1 = MultiHeadAttention(n_head=2, d_model=16, d_k=64, d_v=64, dropout=0.1)
        self.cross_attn2 = MultiHeadAttention(n_head=2, d_model=16, d_k=64, d_v=64, dropout=0.1)

        self.encoder = Conv1DNet(32,out_dim=128,T=self.T_a)
        self.conv = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3)



    def forward(self,x,cond=None):
        # x: b c t
        bs,c,t=x.shape
        x= self.conv(x)
        x =x.permute(0,2,1)
       
        if self.use_cond:
            # temperature embedding
            k_t= self.kernel_t(cond[:,0:-2]) # b,c,t
            k_t = k_t.permute(0,2,1)
       
            # srad embedding
            k_r= self.kernel_r(cond[:,-1].unsqueeze(1))
            k_r = k_r.permute(0,2,1)

            # CRM
            cross_f_1,attn1= self.cross_attn1(k_r,x,x)
            cross_f_2,attn2= self.cross_attn2(k_t,x,x)

            f_cat = torch.cat([cross_f_1,cross_f_2],dim=2)

            x=self.encoder(f_cat.permute(0,2,1))  #x in :b c t
            
        f_out = x
        x = self.classifier(f_out.flatten(1))
        
        return x,f_out.flatten(1),attn1,attn2
    



    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(config, **kwargs):
    model = CropANet(config, **kwargs)
    model.init_weights()
    return model
