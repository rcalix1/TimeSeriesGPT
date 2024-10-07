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
        self.the_offset     = ''
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.excel_matrix   = np.zeros( (250, 30) )

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

    
    def plots_inference_one( self,  l_real, l_pred, yellow_l_SI_data_pred, si_2_all_real_24 ):
        yellow_l_SI_data_pred = np.array( yellow_l_SI_data_pred )
        x = [ i for i in range(   len(l_real)   ) ] 
        plt.axvline(x = 4,  color = 'b') 
        plt.axvline(x = 9,  color = 'b') 
        plt.axvline(x = 14, color = 'b') 
        plt.scatter(x, l_real, label = "real", color='red') 
        plt.plot(   x, l_real, label = "real", color='red')  
        l_pred = np.roll(l_pred, -1)
        plt.scatter(x, l_pred, label = "pred", color='blue') 
        plt.plot(   x, l_pred, label = "pred", color='blue') 
        plt.plot(   x, yellow_l_SI_data_pred, label = "pred1cast",     color='green')   ## green
        plt.plot(   x,      si_2_all_real_24, label = "real SI",       color='red'  )   ## green
        for i in range(l_real.shape[0]):
            j = self.get_j( self.the_offset )
            self.excel_matrix[ self.the_offset+i, j  ] =  l_real[i].round(decimals=2)        ## np.round(l_real[i], 2)  ## deltas
            self.excel_matrix[ self.the_offset+i, j+1] =  l_pred[i].round(decimals=2)        ## np.round(l_pred[i], 2)  ## deltas
            self.excel_matrix[ self.the_offset+i, j+2] =      si_2_all_real_24[i].round(decimals=2) ## full SI
            self.excel_matrix[ self.the_offset+i, j+3] =  yellow_l_SI_data_pred[i].round(decimals=2) ## Full SI
        plt.legend() 
        plt.show()
        

    def plots_inference_two(  self, l_real, l_pred,  yellow_l_SI_data_pred ):
        np_yellow_l_SI_data_pred = np.array( yellow_l_SI_data_pred )
        x       = [ i for i in range(     len(    l_real    )    )  ] 
        

    def plots_to_excel( self, l_real, l_pred,  yellow_l_SI_data_pred, real_SI ):
        xxx = [ i for i in range( l_SI_data_real.shape[0] )]
        plt.title("The excel data")
        plt.plot(   xxx, l_real, label = "delta real ",         color='red'  )   ## red
        plt.plot(   xxx, l_pred, label = "delta pred ",         color='blue'  )   ## red
        plt.legend() 
        plt.show()


    def GPT_get_batch_test( self, test_data ):
        ## x_time_series = torch.tensor(test_data.values).float()       ## pandas to torch
        x_time_series = test_data
        x  = torch.stack(   [   x_time_series[ 0 : -1    ]    ]    ) 
        y  = torch.stack(   [   x_time_series[ 1 :       ]    ]    )
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def func_zh_infer_on_train_set( self, model, history_GPT, x_means, x_standard_devs, data_zh ):
        xb_test, yb_test = self.GPT_get_batch_test( data_zh )
        print(" inference on train set all 105? ")
        print( xb_test.shape )
        print( yb_test.shape )      ## 1, 104, 26
        input_test_x = xb_test[ :,  : 5 ]         ## give first 4 in sequence for GPT to generate the rest
        pred_20_seq = model.generate( input_test_x, 101 )
        y_pred_gpt     = pred_20_seq.detach().cpu().numpy() 
        y_real_gpt     =     yb_test.detach().cpu().numpy() 
        ## xb_real_gpt    =     xb_test.detach().cpu().numpy().squeeze(0)
        y_pred_gpt = y_pred_gpt.squeeze(0)
        y_real_gpt = y_real_gpt.squeeze(0)
        y_pred_gpt = y_pred_gpt[ :-2,  :]
        print( y_pred_gpt.shape )
        print( y_real_gpt.shape )
        xxx = [ i for i in range( y_pred_gpt.shape[0] )]
        for j in range(      y_pred_gpt.shape[1]     ):
            plt.title("preds on the train set" + str(j)     )
            temp = y_pred_gpt[:, j]
            temp = np.roll(temp, -1)
            plt.plot(   xxx, y_real_gpt[:, j], label = "real ",         color='red'  )   ## red
            plt.plot(   xxx, temp            , label = "pred ",         color='blue' )   ## red
            plt.legend() 
            plt.show()
        return 0

    def save_results_to_file( self, results_string, tsGPT_obj ):
        results_string   = results_string + "," + str( tsGPT_obj.the_offset ) + "," 
        results_string   = results_string + "SlidWind1" + "," + str( tsGPT_obj.max_iters ) + "," 
        results_string   = results_string + tsGPT_obj.comment_exp +  "\n"
        exp_results_file = open('experiment_results_file_GPT_CIVS.txt', 'a')
        exp_results_file.write( results_string ) 
        exp_results_file.close()
        np.savetxt("for_excel_7_window.csv", self.excel_matrix, delimiter=",")


    def GPT_generate_inference(self, model, train_data, test_CIVS, x_means, x_standard_devs ):
        frames           = [ train_data[ -5: ], test_CIVS ]
        test_CIVS_concat = pd.concat( frames )
        test_CIVS_tr = torch.tensor(test_CIVS_concat.values).float()
        test_CIVS_tr_scaled = ( test_CIVS_tr - x_means ) / x_standard_devs
        x_test = test_CIVS_tr_scaled
        xb_test, yb_test = self.GPT_get_batch_test( x_test )
        input_test_x = xb_test[ :,  : 5 ]         ## give first 4 in sequence for GPT to generate the rest
        pred_20_seq = model.generate( input_test_x, 14 )
        y_pred_gpt     = pred_20_seq.detach().cpu().numpy() 
        y_real_gpt     =     yb_test.detach().cpu().numpy() 
        xb_real_gpt    =     xb_test.detach().cpu().numpy().squeeze(0)
        y_pred_gpt = y_pred_gpt.squeeze(0)
        y_real_gpt = y_real_gpt.squeeze(0)
        l_real                 = y_real_gpt[ :, 0 ]   
        l_pred                 = y_pred_gpt[ :, 0 ]
        l_real_all_24_features = y_real_gpt[ :, :]
        l_pred_all_24_features = y_pred_gpt[ :, :]
        si_mean                         = x_means[0, 0].numpy()
        si_standard_dev                 = x_standard_devs[0, 0].numpy()
        si_mean_all_24_features         = x_means[0, :].numpy()
        si_standard_dev_all_24_features = x_standard_devs[0, :].numpy()
        l_pred                 = l_pred                   * si_standard_dev                   + si_mean
        l_real                 = l_real                   * si_standard_dev                   + si_mean
        l_pred_all_24_features = l_pred_all_24_features   * si_standard_dev_all_24_features   + si_mean_all_24_features
        l_real_all_24_features = l_real_all_24_features   * si_standard_dev_all_24_features   + si_mean_all_24_features
        xb_real_gpt            = xb_real_gpt              * si_standard_dev_all_24_features   + si_mean_all_24_features
        results_string = self.metrics_function_all_details(  l_pred, l_real,  l_pred_all_24_features, l_real_all_24_features  )
        yellow_l_SI_data_pred = []
        for i in range( len(l_pred) ):
            if (i-1) < 0:
                prev_cast = xb_real_gpt[0, 2] 
            else:
                prev_cast = l_real_all_24_features[i-1, 2]
                ## prev_cast = xb_real_gpt[i, 2] 
                print("this ..." , ( xb_real_gpt[i, 2]  -  l_real_all_24_features[i-1, 2] )  )
            the_curr_val =  prev_cast + l_pred[i]
            yellow_l_SI_data_pred.append( the_curr_val )
        self.plots_inference_one(  l_real, l_pred,  yellow_l_SI_data_pred, l_real_all_24_features[:, 2]  )
        self.plots_inference_two(  l_real, l_pred,  yellow_l_SI_data_pred )
        return results_string 
        

    def function_test_rc( self, train_data, test_CIVS, si_GPT, x_means, x_standard_devs):
        results_string = self.GPT_generate_inference( si_GPT, train_data, test_CIVS, x_means, x_standard_devs  )
        return results_string

    
    def printName(self):
        print( self.MyName  )










