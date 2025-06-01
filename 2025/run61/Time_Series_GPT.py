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


class ResidualSIHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(64, 64)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(64, 1)

        # If input_dim ≠ 64, project for residual addition
        self.input_proj = nn.Linear(input_dim, 64) if input_dim != 64 else nn.Identity()

    def forward(self, x):
        x0 = self.input_proj(x)
        x1 = self.dropout1(self.act1(self.fc1(x)))
        x2 = self.dropout2(self.act2(self.fc2(x1)))
        x_res = x2 + x0
        out = self.fc3(x_res)
        return out


##################################################################


class ResidualInitialProjection(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(256, 512)

        # If input_dim ≠ 64, project for residual addition
        self.input_proj = nn.Linear(input_dim, 256) if input_dim != 256 else nn.Identity()

    def forward(self, x):
        x0 = self.input_proj(x)
        x1 = self.dropout1(self.act1(self.fc1(x)))
        x2 = self.dropout2(self.act2(self.fc2(x1)))
        x_res = x2 + x0
        out = self.fc3(x_res)
        return out
    

##################################################################


class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        
        self.key   = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]
        self.query = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]
        self.value = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]

        ####################
        
        tril_def = torch.tril( torch.ones(block_size, block_size) )  ## [40, 40]
        
        ## tril_def = torch.tril( torch.ones(block_size + 1, block_size + 1))   ## Titans 6.0
        
        ## first is the original (go back to this), second is to focus on first 3 preds, maybe
        ## tril_def = torch.zeros(block_size, block_size) ## remove this
        ## tril_def[:, :4] = 1     ## remove this
        #####################
        

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


###################################################################

class ConvFeatureExtractor(nn.Module):
    
    def __init__(self, input_dim, output_dim, kernel_size=3):
        
        super(ConvFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size//2)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        # Reshape input for Conv1d: [batch, vector_size, sequence_length]
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        # Reshape back to [batch, sequence_length, output_dim]
        return x.permute(0, 2, 1)


##################################################################

   



class Time_Series_GPT(nn.Module):
    
    def __init__(self, params_obj):
        
        super().__init__()
        self.output_size  = params_obj.output_size      ## 35
        self.seq_length   = params_obj.seq_length       ## 10
        self.input_size   = params_obj.input_size       ## 35
        self.num_features = params_obj.num_features     ## 35
        self.device       = params_obj.device
        self.block_size   = params_obj.block_size       ## 10
        self.MyName       = "The GPT model"

        self.pos_emb_table   = nn.Embedding(params_obj.block_size, params_obj.n_embd)      ## [block, 512] or [40, 512]
        
        self.blocks = nn.Sequential(
                *[  Block(params_obj.n_embd, params_obj.n_head, params_obj.block_size) for _ in range(params_obj.n_layer)  ]
        )
        
        self.ln_f        = nn.LayerNorm(  params_obj.n_embd    )        
        self.lm_ffw_head = nn.Linear(params_obj.n_embd, self.num_features)      ## [512, 65] # FFW Layer

        #######################################################################

        self.map_24_512 = nn.Linear(self.num_features, 512)  ## [35, 512] # projection
        self.map_act    = nn.ReLU()

        #######################################################################

        self.map_24_512_1 = nn.Linear(self.num_features, 100)  ## [35, 512] # projection
        self.map_act1     = nn.ReLU()
        self.LayerNorm1   = nn.LayerNorm( 100 )
        
        self.map_24_512_2 = nn.Linear(100, 200)      
        self.map_act2     = nn.ReLU()
        self.LayerNorm2   = nn.LayerNorm( 200 )
        
        self.map_24_512_3 = nn.Linear(200, 512)      
        
        self.dropout_24_512 = nn.Dropout(0.2)
        
        ########################
        
        self.head_si = ResidualSIHead(input_dim=self.num_features)
        
        ########################
        
        self.initial_projection = ResidualInitialProjection(input_dim=self.num_features)
        
        ########################
        
        self.patch_embed = nn.Conv1d(
            in_channels  = 35,
            out_channels = 512,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1
        )

    
        

   

    def forward(self,  idx, targets=None, reasoning_steps=1, backprop_through_steps=False, return_si=False):
        
        ## reasoning_steps (int): Number of reasoning iterations. If 1, use normal mode.
        
        B = idx.shape[0]       ## 32 batch 
        T = idx.shape[1]       ## 40 or 10

        # Clone to avoid modifying original input during reasoning
        ## working_idx = idx.clone()
        working_idx = idx
        
        ############################################################
        
        for step in range(reasoning_steps):
            
            # Crop for causal context if needed, from 1..21, removes 1, so that 1..20
            idx_cond = working_idx[:, -self.block_size:, :]     ## makes sure just block size
            
            ############################################
            '''
            x = self.map_24_512_1(idx_cond)   ## (32, 10, 35) goes in
            x = self.map_act1(x)
            x = self.dropout_24_512(x)
            x = self.LayerNorm1(x)
            
            x = self.map_24_512_2(x)
            x = self.map_act2(x)
            x = self.dropout_24_512(x)
            x = self.LayerNorm2(x)
            
            x = self.map_24_512_3(x)
            '''
            
            ###########################################
            
            x =  self.initial_projection(  idx_cond   )
            
            ###########################################
            
            ## x = idx_cond.permute(0, 2, 1)    ## [B, 35,  T]
            ## x = self.patch_embed(x)          ## [B, 512, T]
            ## x = x.permute(0, 2, 1)           ## [B, T, 512]
            
            
            ###########################################
            
            pos_emb = self.pos_emb_table(torch.arange(T, device=self.device) )
            
            x = x + pos_emb       ## (B, 10, 512)
            
            x      = self.blocks(x)      ## (B, T, E)
            x      = self.ln_f(x)        ## (B, T, E)   ## normalization
            logits = self.lm_ffw_head(x)     ## [B, 40, 65]  or [B, 10, 35] 
            
          
            '''
            ## last_pred = logits[:, -1:, :]  ## shape: (32, 1, 35)
            if reasoning_steps > 1 and step < reasoning_steps - 1:
                if backprop_through_steps:
                    working_idx = logits
                    ## torch.cat((working_idx, last_pred), dim=1)
                else:
                    ## Detach to stop gradient unless we want full backprop (optional)
                    working_idx = logits
                    ## torch.cat((working_idx, last_pred.detach()), dim=1)
            '''

        
        
        ############################################################
       
        ## final_pred = last_pred.squeeze(1)  # shape: (B, F)
        
        ## logits is [B, 10, 35]
        pred_si = self.head_si( logits )  ## .squeeze(-1)  # [B, 10, 1]


        ## replace si HEAD val with f2
        
        pred_si = logits[:, :, 2].unsqueeze(2)     ## makes SI head same as f2. REMOVE
        
        if return_si:
            return logits, pred_si    ## [B, 10, 35], [b, 10, 1]
        else:
            return logits

 
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=14, reasoning_steps=1):
       
        self.eval()
        si_outputs = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:, :]  # trim context if needed
        
            full_tokens_full, next_token_si = self(
                idx_cond,
                reasoning_steps=reasoning_steps,
                backprop_through_steps=False,
                return_si=True 
            )  # next_token_full: (B, F), next_token_si: (B, 1)

            next_token_full = full_tokens_full[:, -1:, :]                 ## [1, 1, 32]
            next_token_si   =    next_token_si[:, -1:, :].unsqueeze(1)    ## [1, 1, 1]
            
            # Use the SI head value to replace the full vector's SI dimension
            ## next_token_full[:, :, 2] = next_token_si   ##.squeeze(-1)          ## remove


            idx = torch.cat((idx, next_token_full), dim=1)  # (B, T+1, F)
            
            si_outputs.append(next_token_si)

        si_preds = torch.cat(si_outputs, dim=1)  # (B, max_new_tokens, 1)
        return idx, si_preds
        
    
    
   



