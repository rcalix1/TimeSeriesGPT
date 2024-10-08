## Author: Ricardo A. Calix, Ph.D.
## Last update Oct 1, 2024
## Released as is with no warranty
## MIT License

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
from datetime import datetime

############################################################

import Time_Series_GPT as Time_Series_GPT

############################################################

class ParamsGPT:

    def __init__(self):
        self.MyName = 'tsGPT'
        self.x_means      = None
        self.x_deviations = None
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size   = 15           ##   40      ## N tokens in sequence
        self.batch_size   = 16 
        self.max_iters    = 200   ## 3000   ## 5000
        self.eval_interval  = 200     
        self.learning_rate  = 0.0001
        self.eval_iters     = 300
        self.vocab_size     = 88  ## 65
        ## every id for a given token is embedded to vector of this size
        self.n_embd            = 512       ## 24        ## 512       ## 24 for time series, silicon is 0           
        self.n_head            = 8         ## 8 attention heads
        self.n_layer           = 6         ## 6 eoncoder layers
        self.dropout           = 0.2
        self.seq_length        = self.block_size 
        self.num_features      = 27        ## from delta si to coke rate
        self.input_size        = self.num_features 
        self.output_size       = self.num_features 
        self.results_string    = ''
        ###########################################
        ## sliding window
        self.for_RNN_data_CIVS = None
        self.comment_exp       = "None"
        self.training_chunk    = 105
        self.length_n          = '' ## int( for_RNN_data_CIVS.shape[0] )
        self.the_range         = self.training_chunk + self.block_size
        self.index_to_slice    = 436
        self.excel_matrix      = np.zeros( (250, 30) )
        self.excel_for_rsquare = np.zeros( (250, 10) )
        self.the_offset        = 0                      ## 0, 15, 30, 45, 60, 75, 90, 105

       

    def printName(self):
        print( self.MyName  )

    ## Shift and create batches
    def get_batch( self, data_gpt ):
        ix = torch.randint(   len(data_gpt) - self.block_size, (self.batch_size,)   )
        x  = torch.stack(    [  data_gpt[   i   : i+self.block_size    ]   for i in ix ]    ) 
        y  = torch.stack(    [  data_gpt[   i+1 : i+1+self.block_size  ]   for i in ix ]    )
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def GPT_get_batch_test( self, test_data ):
        ## x_time_series = torch.tensor(test_data.values).float()       ## pandas to torch
        x_time_series = test_data
        x  = torch.stack(   [   x_time_series[ 0 : -1    ]    ]    ) 
        y  = torch.stack(   [   x_time_series[ 1 :       ]    ]    )
        x, y = x.to(self.device), y.to(self.device)
        return x, y


    def standardize_x_scales( self, data_gpt_pd ):
        data_gpt = torch.tensor(data_gpt_pd.values).float()
        epsilon = 0.0001
        ## print( data_gpt.shape)        
        self.x_means      = data_gpt.mean(0,  keepdim=True)
        self.x_deviations = data_gpt.std( 0,  keepdim=True) + epsilon
        ## x_train_tr_scaled = (x_train_tr - x_means) / x_deviations
        ## y_test_tr_scaled  = (y_test_tr  - y_means) / y_deviations
        return data_gpt, self.x_means, self.x_deviations

    def random_4_runs( self ):
        self.index_to_slice    = 436
        self.training_chunk    = 400
        self.the_range = self.training_chunk + self.block_size
        random.seed( datetime.now().timestamp() )
        self.index_to_slice = random.randrange(0, self.length_n - self.the_range)
        sliced_chunk_CIVS = self.for_RNN_data_CIVS[ self.index_to_slice : self.index_to_slice + self.the_range]
        n = self.block_size
        train_CIVS       = sliced_chunk_CIVS[   : -n ] 
        test_CIVS        = sliced_chunk_CIVS[ -n:   ]
        chunk300to400_train = train_CIVS[ 300  :  ] 
        chunk200to400_train = train_CIVS[ 200  :  ]  
        chunk100to400_train = train_CIVS[ 100  :  ] 
        chunk000to400_train = train_CIVS[      :  ]
        return chunk300to400_train, chunk200to400_train, chunk100to400_train, chunk000to400_train, test_CIVS
      
    def save_file_random_4_runs( self, run_n, data_range, results_string ):
        results_string = results_string + "," + str(run_n) + "," + data_range + "," + str(self.max_iters) + "," + self.comment_exp +  "\n"
        exp_results_file = open('experiment_results_file_GPT_CIVS.txt', 'a')
        exp_results_file.write( results_string ) 
        exp_results_file.close()

    def slidingWindowTrain(self, selec_offset):
        self.the_offset       = selec_offset   
        self.index_to_slice   = 436
        my_index_to_slice     = self.index_to_slice + self.the_offset
        sliced_chunk_CIVS     = self.for_RNN_data_CIVS[ my_index_to_slice : my_index_to_slice + self.the_range]
        n                     = self.block_size
        train_CIVS            = sliced_chunk_CIVS[   : -n ] 
        test_CIVS             = sliced_chunk_CIVS[ -n:   ]
        chunk_slideWind_train = train_CIVS[  :  ] 
        return chunk_slideWind_train, test_CIVS

       

    def plot_losses_training( self, history_GPT ):
        fig, ax = plt.subplots(2, 1) 
        ax[0].set_title(f'GPT  Train  Loss  per epoch')
        ax[0].plot(history_GPT['loss'],      label='all',        color='blue'      )
        ax[0].legend()
        ax[1].set_title(f"weighted losses")
        ax[1].plot(history_GPT['loss_A'],      label='0-5',        color='blue'     )
        ax[1].plot(history_GPT['loss_B'],      label='5-10',       color='red'      )
        ax[1].plot(history_GPT['loss_C'],      label='10-15',      color='green'    )
        fig.tight_layout()
        ax[1].legend(); plt.show()


########################################################################


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe



########################################################################


class moving_avg(nn.Module):

    def __init__(self, x):
        super().__init__()
        self.kernel_size = x.shape[2]
        self.stride      = 1
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        x_avg = self.avg( x )         ## x.permute(0, 2, 1))
        ## x = x.permute(0, 2, 1)
        x = x + x_avg
        return x

###########################################################################
































