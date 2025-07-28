from layers.db2vit import MultiHeadDaubechiesBlock
import torch
import torch.nn as nn
from layers.GeneralEncoders import Encoder, EncoderT
from layers.Embed import DataEmbedding_inverted, DataEmbedding_TimeFocus, DataEmbedding_Simple
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
        self.bn1 = torch.nn.BatchNorm1d(configs.d_model,affine=True)
        self.bn2 = torch.nn.BatchNorm1d(configs.d_model,affine=True)
        
        self.encoder = Encoder(
            [
                MultiHeadDaubechiesBlock(
                            dim=configs.d_model,
                            ffn_dim=768,
                            levels=configs.levels,
                            heads=8
                            ) for l in range (configs.e_layers)
            ], d_model=configs.d_model
        )
    # a = self.get_parameter_number()
    #
    # def get_parameter_number(self):
    #     """
    #     Number of model parameters (without stable diffusion)
    #     """
    #     total_num = sum(p.numel() for p in self.parameters())
    #     trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     trainable_ratio = trainable_num / total_num
    #
    #     print('total_num:', total_num)
    #     print('trainable_num:', total_num)
    #     print('trainable_ratio:', trainable_ratio)
    
    def get_attention_masks(self, input):
        _ , seq_len, _ = input.shape
        return torch.ones((1,1,seq_len,seq_len)).to(torch.bool)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        # enc_out = self.bn1(enc_out.permute(0,2,1)).permute(0,2,1)
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        # print(f'the shape of the encoder input is: {enc_out.shape}')
        enc_out = self.encoder(enc_out, attn_mask=None)
        # print(f'the shape of the encoder output is: {enc_out.shape}')
        # enc_out = self.bn2(enc_out.permute(0,2,1)).permute(0,2,1)
        
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        # print(f'the shape of the decoder output is: {dec_out.shape}')

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :] # [B, L, D]