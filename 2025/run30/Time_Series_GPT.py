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


class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        
        self.key   = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]
        self.query = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]
        self.value = nn.Linear(n_embd, head_size, bias=False)  ## [512, 64]

        tril_def = torch.tril( torch.ones(block_size, block_size) )  ## [40, 40]
        ## tril_def = torch.tril( torch.ones(block_size + 1, block_size + 1))   ## Titans 6.0
        
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


class Time_Series_GPT_old(nn.Module):
    
    def __init__(self, params_obj):
        
        super().__init__()
        self.output_size   = params_obj.output_size
        self.seq_length    = params_obj.seq_length       ## 40 or 15
        self.input_size    = params_obj.input_size
        self.num_features  = params_obj.num_features
        self.device        = params_obj.device
        self.block_size    = params_obj.block_size
        self.embd          = params_obj.n_embd
        self.fft_freq_size = 0
        self.MyName        = "The GPT model"

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

        ########################################################################

        # Instantiate convolutional feature extractor
        self.conv_extractor = ConvFeatureExtractor(input_dim=self.input_size, output_dim=self.embd , kernel_size=3)

        ########################################################################

        self.freq_embedding_layer = nn.Linear(   18, self.embd   )
        

     



    def forward(self,  idx, targets):

        
        frequency_features   = torch.fft.rfft( idx, dim=-1).abs() 
        ## print( frequency_features.shape ) 
        freq_embeds          = self.freq_embedding_layer( frequency_features  )  # [batch, seq_len, embedding_dim]
        ## print( freq_embeds.device ) 

        ############################################################

        conv_features = self.conv_extractor(idx)         # [batch, seq_len, embedding_dim]

        ############################################################

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

        ###########################################################
        
        tok_emb = idx                        ## (B, 15, 512)

        ###########################################################
        
        pos_emb = self.pos_emb_table( torch.arange(T, device=self.device) )  
        
        ###########################################################
        
        ## x = tok_emb + pos_emb + conv_emb + per_conv_emb    ## [B, T, E] or [N, 40, 512], now [N, 15, 24]

      

        x = tok_emb + pos_emb + conv_features + freq_embeds



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




    
#########################################



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
        
        ## self.pos_emb_table   = nn.Embedding(params_obj.block_size + 1, params_obj.n_embd) ## Titans 1.0        
        ## self.memory_token    = nn.Parameter(torch.randn(1, 1, params_obj.n_embd)) # assuming 512 embed dim # Titans 2.0

        
        self.blocks = nn.Sequential(
                *[  Block(params_obj.n_embd, params_obj.n_head, params_obj.block_size) for _ in range(params_obj.n_layer)  ]
        )
        
        self.ln_f        = nn.LayerNorm(  params_obj.n_embd    )        
        self.lm_ffw_head = nn.Linear(params_obj.n_embd, self.num_features)      ## [512, 65] # FFW Layer

        #######################################################################

        self.map_24_512 = nn.Linear(self.num_features, 512)  ## [24, 512] # projection
        self.map_act    = nn.ReLU()

        #######################################################################

        self.map_24_512_1 = nn.Linear(self.num_features, 100)  ## [24, 512] # projection
        self.map_act1     = nn.ReLU()
        self.LayerNorm1   = nn.LayerNorm( 100 )
        
        self.map_24_512_2 = nn.Linear(100, 200)      ## [24, 512] # projection
        self.map_act2     = nn.ReLU()
        self.LayerNorm2   = nn.LayerNorm( 200 )
        
        self.map_24_512_3 = nn.Linear(200, 512)      ## [24, 512] # projection
        
        self.dropout_24_512 = nn.Dropout(0.2)
        
        ########################
        
   
        self.head_si_old = nn.Sequential(
            nn.Linear(self.num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
         )
    
        ########################
        
        self.head_si = ResidualSIHead(input_dim=self.num_features)

    
        

   

    def forward(self,  idx, targets=None, reasoning_steps=1, backprop_through_steps=False, return_si=False):
        
        ## targets (Tensor or None): Optional targets for loss calculation
        ## reasoning_steps (int): Number of reasoning iterations. If 1, use normal mode.
        ## logits: Final prediction (B, F)
        ## loss (optional): If targets provided, return MSE loss
        
        B = idx.shape[0]       ## 16 batch 
        T = idx.shape[1]       ## 40 or 15

        # Clone to avoid modifying original input during reasoning
        working_idx = idx.clone()
        
        ############################################################
        
        for step in range(reasoning_steps):
            
            # Crop for causal context if needed, from 1..21, removes 1, so that 1..20
            idx_cond = working_idx[:, -self.block_size:, :]     ## makes sure just block size, 
            
            # === Begin Core Forward ===
            # Projection MLP
            
            x = self.map_24_512_1(idx_cond)   ## (8, 15, 24) goes in
            x = self.map_act1(x)
            x = self.dropout_24_512(x)
            x = self.LayerNorm1(x)
            
            x = self.map_24_512_2(x)
            x = self.map_act2(x)
            x = self.dropout_24_512(x)
            x = self.LayerNorm2(x)
            
            x = self.map_24_512_3(x)
            
            
            ##########################
            ## Titans 3.0
            ## B   = x.size(0)
            ## mem = self.memory_token.repeat(B, 1, 1)  # (B, 1, D)
            ## x   = torch.cat([mem, x], dim=1)  # (B, T+1, D)
            ## T = T+1
            ## T = x.shape[1]  # update T dynamically
            ##########################
            
            pos_emb = self.pos_emb_table(torch.arange(T, device=self.device) )
            ## pos_emb = self.pos_emb_table(torch.arange(T, device=self.device) )   ## Titans 4.0
            
            ## x = tok_emb + pos_emb + conv_emb + per_conv_emb  ## [B, T, E] or [N, 40, 512], now [N, 15, 24]
            ## x = tok_emb + pos_emb 
            
            x = x + pos_emb       ## (B, 15, 512)
            
            
                       
            # Transformer blocks
            x = self.blocks(x)      ## (B, T, E)
            x = self.ln_f(x)        ## (B, T, E)   ## normalization
            logits = self.lm_ffw_head(x)     ## [B, 40, 65]  or [N, 15, 24] 
            # === End Core Forward ===
            
            # Extract latest step’s prediction
            last_pred = logits[:, -1:, :]  # shape: (B, 1, F)
            
            ## last_pred = logits[:, 0:1, :]  # Use memory token output ## Titans 5.0 remove, Head section
            ## Titans 6.0 , go to the Head class an remove in init func 
            

            # In reasoning mode, feed it back in
            if reasoning_steps > 1 and step < reasoning_steps - 1:
                
                if backprop_through_steps:
                    working_idx = torch.cat((working_idx, last_pred), dim=1)
                else:
                    ## Detach to stop gradient unless we want full backprop (optional)
                    working_idx = torch.cat((working_idx, last_pred.detach()), dim=1)
            

        ############################################################
       
        final_pred = last_pred.squeeze(1)  # shape: (B, F)
        

        pred_si = self.head_si(final_pred).squeeze(-1)  # [B]
        ## pred_si = torch.clamp(pred_si, min=0.5, max=1.1)  # Adjust bounds if your SI values differ

        
        ## return final_pred
        ## return final_pred, pred_si  
    
        if return_si:
            return final_pred, pred_si
        else:
            return final_pred

        
       
        


    '''        
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
    '''
    
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=14, reasoning_steps=1):
        """
        Autoregressive generation with reasoning steps.
        idx: (B, T, F) input sequence
        Returns: (B, T + max_new_tokens, F)
        """
        self.eval()
        si_outputs = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:, :]  # trim context if needed
        
            next_token_full, next_token_si = self(
                idx_cond,
                reasoning_steps=reasoning_steps,
                backprop_through_steps=False,
                return_si=True 
            )  # next_token_full: (B, F), next_token_si: (B, 1)


            next_token_full = next_token_full.unsqueeze(1)  # (B, 1, F)
            next_token_si   = next_token_si.unsqueeze(1)    # (B, 1, 1)


            # Append prediction to input sequence
            idx = torch.cat((idx, next_token_full), dim=1)  # (B, T+1, F)
            
            si_outputs.append(next_token_si)

        si_preds = torch.cat(si_outputs, dim=1)  # (B, max_new_tokens, 1)
        return idx, si_preds
        
    
    
   



