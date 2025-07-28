import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class EncoderT(nn.Module):
    def __init__(self, attn_layers):
        super(EncoderT, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None):
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for attn_layer in self.attn_layers:
            x, attn_masks, aux_loss = attn_layer([x, attn_mask])
            total_aux_loss += aux_loss
        return x, attn_masks, total_aux_loss
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, d_model):
        super(Encoder, self).__init__()
        self.len_layers = len(attn_layers)
        self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(d_model, affine=True) for _ in range(self.len_layers)
                ])
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None):
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i in range(self.len_layers):
            x = self.attn_layers[i](x)
            x = self.batch_norms[i](x.permute(0,2,1)).permute(0,2,1)
        return x
    