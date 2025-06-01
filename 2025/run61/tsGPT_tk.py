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
import similaritymeasures

############################################################

import Time_Series_GPT as Time_Series_GPT

############################################################

class ParamsGPT:

    def __init__(self):
        self.MyName            = 'tsGPT'
        self.x_means           = None
        self.x_deviations      = None
        self.device            = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size        = 10   ## 15           ##   40      ## N tokens in sequence
        self.batch_size        = 128  ## 64 ## 16 
        self.max_iters         = 1000 ## 1000 ## 3000  ## 3000   ## 200  ## 5000   ## 100 worked fine 
        self.eval_interval     = 100     
        self.learning_rate     = 0.001   # 0.0001
        self.eval_iters        = 300
        self.vocab_size        = 88  
        ## every id for a given token is embedded to vector of this size
        self.n_embd            = 512       ## 24        ## 512       ## 24 for time series, silicon is 0           
        self.n_head            = 8        ## 8 attention heads
        self.n_layer           = 6         ## 6 eoncoder layers
        self.dropout           = 0.1
        self.seq_length        = self.block_size 
        self.num_features      = 35        ## from delta si to coke rate
        self.input_size        = self.num_features 
        self.output_size       = self.num_features 
        self.results_string    = ''
        ###########################################
        ## sliding window
        self.for_RNN_data_CIVS = None
        self.comment_exp       = "None"
        self.training_chunk    = 105     ## 105 or 400 
        self.length_n          = ''      ## 1300  or  int( for_RNN_data_CIVS.shape[0] )
        self.the_range         = self.training_chunk + self.block_size
        self.index_to_slice    = 436
        self.excel_matrix      = np.zeros( (250, 30) )
        self.excel_for_rsquare = np.zeros( (250, 10) )
        self.the_offset        = 0                      ## 0, 15, 30, 45, 60, 75, 90, 105
        self.how_many          = 9

       
    def printName(self):
        print( self.MyName  )

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

    
    def slidingWindowTrain(self, selec_offset ):
        self.index_to_slice   = 436
        random.seed( datetime.now().timestamp() )
        self.index_to_slice = random.randrange(0, self.length_n - self.the_range)
        self.the_offset       = selec_offset   
        self.training_chunk   = 105
        self.the_range        = self.training_chunk + self.block_size
        my_index_to_slice     = self.index_to_slice + self.the_offset
        sliced_chunk_CIVS     = self.for_RNN_data_CIVS[ my_index_to_slice : my_index_to_slice + self.the_range]
        train_CIVS            = sliced_chunk_CIVS[                 : -self.block_size ] 
        test_CIVS             = sliced_chunk_CIVS[ -self.block_size:                  ]
        return train_CIVS, test_CIVS

    def func_SIMs_plot_during_training(self,  a, b  ):
        
        x   = [ i for i in range(   a_np.shape[0]   ) ]  
        plt.plot(   x, b_np, label = "real", color='red')  
        plt.plot(   x, a_np, label = "pred", color='blue') 
        plt.legend() 
        plt.show()


    def func_get_Curve_SIMs(self,  a, b  ):
        a = a[:, :, 2]
        b = b[:, :, 2]
        a_np = a.cpu().detach().numpy()
        b_np = b.cpu().detach().numpy()
        ## print(a_np.shape)
        ## print(b_np.shape)
        Curve_SIMSs = np.zeros( (a_np.shape[0], 6) )
        ######
        for j in range( a_np.shape[0] ):
            a_np_j = a_np[j]
            b_np_j = b_np[j]
            x   = [ i for i in range(   a_np_j.shape[0]   ) ] 
            y   = a_np_j 
            exp_data = np.zeros((a_np_j.shape[0], 2))
            exp_data[:, 0] = x
            exp_data[:, 1] = y
            ######
            y = b_np_j
            num_data  = np.zeros((a_np_j.shape[0], 2))
            num_data[:, 0] = x
            num_data[:, 1] = y

            # quantify the difference between the two curves using PCM
            pcm = similaritymeasures.pcm(                     exp_data, num_data)
            df = similaritymeasures.frechet_dist(             exp_data, num_data)
            area = similaritymeasures.area_between_two_curves(exp_data, num_data)
            dtw, d = similaritymeasures.dtw(                  exp_data, num_data)
            mae = similaritymeasures.mae(                     exp_data, num_data)
            mse = similaritymeasures.mse(                     exp_data, num_data)
            ## cl = similaritymeasures.curve_length_measure(     exp_data, num_data)
            
            ## print(pcm)
            Curve_SIMSs[ j, 0] = pcm    ## rank 1
            Curve_SIMSs[ j, 1] = df
            Curve_SIMSs[ j, 2] = area
            Curve_SIMSs[ j, 3] = dtw    ## rank 2
            Curve_SIMSs[ j, 4] = mse
            Curve_SIMSs[ j, 5] = mae
            ## Curve_SIMSs[ j, 6] = cl

        ## Curve_SIMSs            = Curve_SIMSs / np.linalg.norm( Curve_SIMSs, axis=0, keepdims=True )  ## remove rc
        norm_Curve_SIMSs_means    = np.mean(  Curve_SIMSs, axis=0 )
        return norm_Curve_SIMSs_means

    
    def MySoftmax(self, x):
        return( np.exp(x)/np.exp(x).sum() )
    
       
    def plot_losses_training( self, history_GPT ):
        fig, ax = plt.subplots(2, 1) 
        ax[0].set_title(f'GPT  Train  Loss  per epoch')
        ax[0].plot(history_GPT['loss'],      label='all',        color='blue'      )
        ax[0].legend()
        ax[1].set_title(f"weighted losses")
        ax[1].plot(history_GPT['loss_A'],      label='0-5',        color='blue'     )
        ax[1].plot(history_GPT['loss_B'],      label='5-10',       color='red'      )
        ax[1].plot(history_GPT['loss_C'],      label='10-15',      color='green'    )
        ax[1].plot(history_GPT['loss_SIM0'],   label='loss_SIM0',  color='gold'    )
        ax[1].plot(history_GPT['loss_SIM1'],   label='loss_SIM1',  color='black'    )
        ax[1].plot(history_GPT['loss_SIM2'],   label='loss_SIM2',  color='purple'    )
        ax[1].plot(history_GPT['loss_SIM3'],   label='loss_SIM3',  color='orange'    )
        ax[1].plot(history_GPT['loss_SIM4'],   label='loss_SIM4',  color='green'    )
        ax[1].plot(history_GPT['loss_SIM5'],   label='loss_SIM5',  color='green'    )
        fig.tight_layout()
        ax[1].legend(); plt.show()









