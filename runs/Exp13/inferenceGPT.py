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


############################################################

class inferenceGPT:

    def __init__(self):
        self.MyName         = 'inferenceGPT'
        self.eval_criterion = nn.MSELoss()
        self.the_offset     = None
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.excel_matrix   = np.zeros( (250, 30) )
        self.train_or_test  = None
        self.how_many       = 0

    def RSE(self, pred, true):
        return np.sqrt( np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2) )

    def MAE(self, pred, true):
        return np.mean(np.abs(pred - true))

    def MSE(self, pred, true):
        return np.mean((pred - true) ** 2)

    def RMSE(self, pred, true):
        return np.sqrt( self.MSE(pred, true) )

    def MAPE(self, pred, true):
        return np.mean(np.abs((pred - true) / true))

    def MSPE(self, pred, true):
        return np.mean(np.square((pred - true) / true))

    def metric(self, pred, true):
        mae  = self.MAE( pred, true)
        mse  = self.MSE( pred, true)
        rmse = self.RMSE(pred, true)
        mape = self.MAPE(pred, true)
        mspe = self.MSPE(pred, true)
        rse  = self.RSE( pred, true)
        return mae, mse, rmse, mape, mspe, rse  

    def metrics_function_all_details(self, l_pred, l_real, l_pred_all_24_features,l_real_all_24_features  ):
        print( l_pred.shape )
        print( l_real.shape )
        mse_eval_bins      = self.eval_criterion( torch.FloatTensor( l_pred ),        torch.FloatTensor( l_real        ) )
        mse_eval_bins0_5   = self.eval_criterion( torch.FloatTensor( l_pred[4:9] ),   torch.FloatTensor( l_real[4:9]   ) )
        mse_eval_bins5_10  = self.eval_criterion( torch.FloatTensor( l_pred[9:14] ),  torch.FloatTensor( l_real[9:14]  ) )
        mse_eval_bins10_15 = self.eval_criterion( torch.FloatTensor( l_pred[14:19] ), torch.FloatTensor( l_real[14:19] ) )
        
        metric_mse_loss_SI_only                = mse_eval_bins.item()
        metric_mae_mse_rmse_mape_mspe_rse_corr = self.metric(    l_pred, l_real ) 
        metric_rsquare_SI_only                 = r2_score(  l_real, l_pred )
        metric_rsquare_all_features            = r2_score( 
                 np.reshape( l_real_all_24_features, (-1) ), 
                 np.reshape( l_pred_all_24_features, (-1) ) 
        ) 
        metric_rsquare_SI_f2_features            = r2_score( 
                 np.reshape( l_real_all_24_features[-self.how_many:, 2], (-1) ), 
                 np.reshape( l_pred_all_24_features[-self.how_many:, 2], (-1) ) 
        ) 
        results_string = "mse_SI_only," + str(round( metric_mse_loss_SI_only, 4)) 
        results_string = results_string + "," + "rsquare_SI_only" + "," + str(round( metric_rsquare_SI_only, 4))
        results_string = results_string + "," + "rsquare_all_features" + "," + str(round( metric_rsquare_all_features, 4))
        several_metrics = str( metric_mae_mse_rmse_mape_mspe_rse_corr ).replace("(", "").replace(")","")
        results_string = results_string + "," + "mae_mse_rmse_mape_mspe_rse"  + "," + several_metrics
        print("Test MSE Loss - SI only: ",        mse_eval_bins.item()         )     ## :.4f }')
        print("Test MSE Loss - SI only 0-5: ",    mse_eval_bins0_5.item()      )     ## :.4f }')
        print("Test MSE Loss - SI only 5-10: ",   mse_eval_bins5_10.item()     )     ## :.4f }')
        print("Test MSE Loss - SI only 10-15: ",  mse_eval_bins10_15.item()    )     ## :.4f }')
        print("mae, mse, rmse, mape, mspe, rse, corr")
        print(    metric_mae_mse_rmse_mape_mspe_rse_corr    )
        print( "Testing R**2 - SI only: ", metric_rsquare_SI_only  )
        print( "Testing R**2 - SI only: ", r2_score(  np.reshape( l_real, (-1) ), np.reshape( l_pred, (-1) )      )  )
        print( "Testing R**2 - All features: ", metric_rsquare_all_features )
        print( "Testing R**2 - (f2) - SI full: ", metric_rsquare_SI_f2_features )
        return results_string 
        

    def get_j( self, the_offset ):
        ## 0, 15, 30, 45, 60, 75, 90, 105
        if the_offset == 0:
            j = 0
        if the_offset == 15:
            j = 4
        if the_offset == 30:
            j = 8
        if the_offset == 45:
            j = 12
        if the_offset == 60:
            j = 16
        if the_offset == 75:
            j = 20
        if the_offset == 90:
            j = 24
        return j
        

    def add_data_to_excel_matrix(self, l_real, l_pred, yellow_l_SI_data_pred, si_2_all_real_24 ):
        for i in range(l_real.shape[0]):
            j = self.get_j( self.the_offset )
            self.excel_matrix[ self.the_offset+i, j  ] =  l_real[i].round(decimals=2)        ## np.round(l_real[i], 2)  ## deltas
            self.excel_matrix[ self.the_offset+i, j+1] =  l_pred[i].round(decimals=2)        ## np.round(l_pred[i], 2)  ## deltas
            self.excel_matrix[ self.the_offset+i, j+2] =  si_2_all_real_24[i].round(decimals=2) ## full SI
            self.excel_matrix[ self.the_offset+i, j+3] =  yellow_l_SI_data_pred[i].round(decimals=2) ## Full SI
        

    def plots_inference_one( self,  l_f0_real, l_f0_pred, yellow_l_SI_data_pred, si_2_all_real_24, l_f2_pred ):

        ## if self.train_or_test:
        ## plt.axvline(x = 4,  color = 'b') 
        ## plt.axvline(x = 9,  color = 'b') 
        plt.axvline(x = 19, color = 'b') 
        
        yellow_l_SI_data_pred = np.array( yellow_l_SI_data_pred )
        x                     = [ i for i in range(   len(l_f0_real)   ) ] 

        l_f0_pred        = np.roll(l_f0_pred,        -1)
        si_2_all_real_24 = np.roll(si_2_all_real_24, +1)
        
        ## plt.scatter(x, l_f0_real, label = "real", color='red') 
        ## plt.plot(   x, l_f0_real, label = "real", color='red')  
        
        ## plt.scatter(x, l_f0_pred, label = "pred", color='blue') 
        ## plt.plot(   x, l_f0_pred, label = "pred", color='blue') 
        
        ## plt.plot(   x, yellow_l_SI_data_pred, label = "pred1cast",     color='green')   
        plt.plot(   x,      si_2_all_real_24, label = "real SI",       color='red'  )   
        plt.plot(   x,             l_f2_pred, label = "GPT pred SI",   color='gold'  ) 
        
        plt.legend() 
        plt.show()

        if not self.train_or_test:
            self.add_data_to_excel_matrix( l_f0_real, l_f0_pred, yellow_l_SI_data_pred, si_2_all_real_24 )


    def GPT_get_batch_test( self, x_time_series  ):
        x  = torch.stack(   [   x_time_series[ 0 : -1    ]    ]    ) 
        y  = torch.stack(   [   x_time_series[ 1 :       ]    ]    )
        x, y = x.to(self.device), y.to(self.device)
        return x, y

                                   
    def prep_data_for_GPT_gen(self, train_data, test_CIVS, x_means, x_standard_devs ):
        if self.train_or_test:
            frames           = [ train_data[ -5: ], test_CIVS ]    ## 5 + 15
            test_CIVS_concat = pd.concat( frames )
        else:
            test_CIVS_concat = train_data[ : 50 ]
        
        test_CIVS_tr        = torch.tensor(test_CIVS_concat.values).float()
        test_CIVS_tr_scaled = ( test_CIVS_tr - x_means ) / x_standard_devs
        return test_CIVS_tr_scaled 
        

    def save_Excel_to_CSV(self):
        excel_matrix_pd = pd.DataFrame( self.excel_matrix )
        excel_matrix_pd.to_csv("for_excel_15_slide_window.csv")
        line = 'id,delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,'
        line = line + 'delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,'
        line = line + 'delta_real,delta_pred,DrZ_real,DrZ_pred,None,None'
        with open("for_excel_15_slide_window.csv", 'r+') as file: 
            file_data = file.read() 
            file.seek(0, 0) 
            file.write(line + '\n' + file_data)
        file.close()
        

    def get_prev_cast_plus_delta(self, l_f0_pred, xb_real_gpt, l_real_all_24_features):
        yellow_l_SI_data_pred = []
        for i in range( len(l_f0_pred) ):
            if (i-1) < 0:
                prev_cast = xb_real_gpt[0, 2] 
            else:
                prev_cast = l_real_all_24_features[i-1, 2]
            ## the_curr_val =  prev_cast + l_f0_pred[i]
            the_curr_val =  xb_real_gpt[i-1, 2]  + l_f0_pred[i]
            yellow_l_SI_data_pred.append( the_curr_val ) 
        return yellow_l_SI_data_pred
        

    def un_scale_pred_real_data(self, the_data, x_means, x_standard_devs ):
        si_mean_all_24_features         =         x_means[0, :].numpy()
        si_standard_dev_all_24_features = x_standard_devs[0, :].numpy()
        data_all_24_features  = the_data.detach().cpu().numpy().squeeze(0)
        data_all_24_features  = data_all_24_features   * si_standard_dev_all_24_features   + si_mean_all_24_features
        return data_all_24_features 

    
    def POST_Process_GPT_inference(self, pred_20_seq, xb_test, yb_test, x_means, x_standard_devs ):
        
        l_pred_all_24_features  = self.un_scale_pred_real_data( pred_20_seq, x_means, x_standard_devs )
        l_f0_pred               = l_pred_all_24_features[ :, 0 ]
        l_f2_pred               = l_pred_all_24_features[ :, 2 ]

        l_real_all_24_features  = self.un_scale_pred_real_data( yb_test, x_means, x_standard_devs )
        l_f0_real               = l_real_all_24_features[ :, 0 ] 

        xb_real_gpt             = self.un_scale_pred_real_data( xb_test, x_means, x_standard_devs )
 
        results_string = self.metrics_function_all_details(  l_f0_pred, l_f0_real,  l_pred_all_24_features, l_real_all_24_features  )
        
        yellow_l_SI_data_pred = self.get_prev_cast_plus_delta( l_f0_pred, xb_real_gpt, l_real_all_24_features)
            
        self.plots_inference_one(  l_f0_real, l_f0_pred,  yellow_l_SI_data_pred, l_real_all_24_features[:, 2], l_f2_pred )
        
        return results_string
      
    
    def function_test_rc( self, train_data, test_CIVS,  model, x_means, x_standard_devs, train_or_test, how_many ):
        self.train_or_test = train_or_test
        self.how_many      = how_many
        x_test = self.prep_data_for_GPT_gen( train_data, test_CIVS, x_means, x_standard_devs )
        xb_test, yb_test = self.GPT_get_batch_test( x_test )   
        ## input_test_x     = xb_test[ :,  : 5 ]         ## give first 4 or 5 in sequence for GPT to generate the rest
        input_test_x     = xb_test[ :,  : 20 ] 
        pred_20_seq      = model.generate( input_test_x, how_many )   ## how_many = 14 or 99
        results_string = self.POST_Process_GPT_inference( pred_20_seq, xb_test, yb_test, x_means, x_standard_devs )
        return results_string 
        
    
    def printName(self):
        print( self.MyName  )









