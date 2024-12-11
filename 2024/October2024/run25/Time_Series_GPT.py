## Author: Ricardo A. Calix, Ph.D.
## Last update Oct 1, 2024
## Released as is with no warranty
## MIT License
## A Time Series GPT

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import sklearn
import random
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
## coefficient of determination 
from sklearn.metrics import r2_score
from einops import rearrange
from math import sqrt, log
torch.manual_seed(256)


##################################################################


class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        
        self.key   = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]
        self.query = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]
        self.value = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]

        tril_def = torch.tril( torch.ones(block_size, block_size) )  ## [40, 40]
        
        self.register_buffer(
                  'tril', 
                  tril_def
               )
        
        self.dropout = nn.Dropout( 0.1 )

    def forward(self, x):
        
        B, T, E = x.shape   ## [batch_size, 40, 512] or [B, 15, 512]
        
        k = self.key(   x )            ## k = (B, T, 64)
        q = self.query( x )            ## q = (B, T, 64)

        E2 = 64     ## I think this is 64 and not 512
        ## (B, T, E) @ (B, E, T)  -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * E2 ** -0.5        
        
        wei = wei.masked_fill(
                      self.tril[:T, :T] == 0, 
                      float('-inf')
        )   
        
        ## (B, T, T)
        wei = F.softmax( wei, dim= -1 )         ## (B, T, T)
        wei = self.dropout(   wei   )
        
        v   = self.value(  x  )   ## x = (B, 40, E)
        out = wei @ v             ## (B, T, T) @ (B, T, 64) -> (B, T, 64)
        
        return out
        


##################################################################


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, block_size):    ## (8, 64)
        super().__init__()
        self.heads = nn.ModuleList(  [ Head(head_size, n_embd, block_size) for _ in range(num_heads) ] )
        self.proj  = nn.Linear(n_embd, n_embd)   ## 512, 512
        self.dropout = nn.Dropout( 0.1 )
    
    def forward(self, x):
        out = torch.cat(   [ h(x) for h in self.heads ], dim = -1   )
        out = self.proj(  out   )
        out = self.dropout(   out   )
        return out

##################################################################

class FeedForward(nn.Module):

    def __init__(self, n_embd):         ## 512
        
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),      ## [512, 4*512]
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),      ## [4*512, 512]
            nn.Dropout( 0.1 ),
        )
        
    def forward(self, x):
        return self.net(x)

##################################################################

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head, block_size):     ## (512, 8)
        super().__init__()
        head_size = n_embd // n_head        ## 64
        self.sa   = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward( n_embd)    ## 512
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(     self.ln1(x)      )
        x = x + self.ffwd(   self.ln2(x)      )
        return x


##################################################################


class Time_Series_GPT(nn.Module):
    
    def __init__(self, params_obj):
        
        super().__init__()
        self.output_size  = params_obj.output_size
        self.seq_length   = params_obj.seq_length       ## 40 or 15
        self.input_size   = params_obj.input_size
        self.num_features = params_obj.num_features
        self.device       = params_obj.device
        self.block_size   = params_obj.block_size
        self.MyName       = "The GPT model"

        self.pos_emb_table   = nn.Embedding(params_obj.block_size, params_obj.n_embd)      ## [block, 512] or [40, 512]
        
        self.blocks = nn.Sequential(
                *[  Block(params_obj.n_embd, params_obj.n_head, params_obj.block_size) for _ in range(params_obj.n_layer)  ]
        )
        
        self.ln_f         = nn.LayerNorm(  params_obj.n_embd    )        
        self.lm_ffw_head  = nn.Linear(params_obj.n_embd, self.num_features)      ## [512, 65] # FFW Layer
        
        self.SI_only_head1         = nn.Linear(self.num_features, 10)
        self.SI_only_act1         = nn.GELU()
        self.SI_only_LayerNorm1   = nn.LayerNorm( 10 )
        self.SI_only_head2        = nn.Linear(10, 1)

        #######################################################################

        self.map_24_512 = nn.Linear(self.num_features, 512)  ## [24, 512] # projection
        self.map_act    = nn.GELU()

        #######################################################################

        self.map_24_512_1 = nn.Linear(self.num_features, 100)  ## [24, 512] # projection
        self.map_act1     = nn.GELU()
        self.LayerNorm1   = nn.LayerNorm( 100 )
        
        self.map_24_512_2 = nn.Linear(100, 200)      ## [24, 512] # projection
        self.map_act2     = nn.GELU()
        self.LayerNorm2   = nn.LayerNorm( 200 )
        
        self.map_24_512_3 = nn.Linear(200, 512)      ## [24, 512] # projection
        
        self.dropout_24_512 = nn.Dropout(0.1)
   

    def forward(self,  idx, targets):

        B = idx.shape[0]       ## 16 batch 
        T = idx.shape[1]       ## 40 or 15

        ############################################################

        idx = self.map_24_512_1( idx )       ## (8, 15, 24) goes in
        idx = self.map_act1(     idx )
        idx = self.dropout_24_512(     idx )
        idx = self.LayerNorm1(   idx )
        
        idx = self.map_24_512_2( idx )
        idx = self.map_act2(     idx )
        idx = self.dropout_24_512(     idx )
        idx = self.LayerNorm2(   idx )
        
        idx = self.map_24_512_3( idx )
        
        tok_emb = idx                        ## (B, 15, 512)

        ###########################################################
        
        pos_emb = self.pos_emb_table( torch.arange(T, device=self.device) )  
        
        ###########################################################
        
        ## x = tok_emb + pos_emb + conv_emb + per_conv_emb    ## [B, T, E] or [N, 40, 512], now [N, 15, 24]

        x = tok_emb + pos_emb 

        ############################################################
        
        x = self.blocks(  x  )               ## (B, T, E)   
        x = self.ln_f(    x  )               ## (B, T, E)   ## norm
        logits  = self.lm_ffw_head(x)         ## [B, 40, 65]  or [N, 15, 24]
        ############################
        SI_only_delta = self.SI_only_head1( logits )        
        SI_only_delta = self.SI_only_act1(  SI_only_delta )        
        SI_only_delta = self.SI_only_LayerNorm1(  SI_only_delta )
        SI_only_delta = self.SI_only_head2( SI_only_delta )
        logits[:, :, 2] = logits[:, :, 2] + SI_only_delta[:, :, 0].detach()
        ############################
        return logits
        
        
    def generate(self, idx, max_new_tokens):    ## idx is (B, T)
        print("max tokens ", max_new_tokens)
        print(idx.shape)
        for _ in range(max_new_tokens):
            ## crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:, :]
            logits = self(idx_cond, 0 )    ## ## get preds
            logits = logits[:, -1, :]    ## focus on last one (B, E)
            logits = logits.unsqueeze(0)
            ## probs = F.softmax(logits, dim= -1)    ## (B, E) get probs
            ## idx_next = torch.multinomial(probs, num_samples=1)     ## (B, 1) selected
            idx = torch.cat(  (idx, logits), dim=1  )   ## (B, T+1) append sample to running sequence
            
        return idx













