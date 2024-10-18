
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

plt.rcParams["figure.figsize"] = [15, 5]
pd.options.display.max_columns = None

############################################################

class tsDataWrangling:
    
    def __init__(self):
        self.MyName                      = 'dataWrangling'
        self.dataFile                    = ''
        self.filename_dates_map          = ''
        self.for_RNN_data_CIVS           = None
        self.delta_for_RNN_data_CIVS     = None
        self.SI_lookup_for_RNN_data_CIVS = None
        self.df_Dates_Map                = None
        self.cols_list_DF                = None

        self.l_new  =      ['SI_f1', 
                            'HOST_BLAST_MOISTURE_f3','HOT_BLAST_TMP_NS_f3','NAT_GAS _INJECTION_f3', 'WINDRATE_f3',
                            'HIGH_PURITY_OXYGEN_f3', 'COAL_FLOW_f3','Cast_Avg_Mn_f2','Slag_Fe_f2',
                            'Selec_Top_Gas_CO_f3','Selec_Top_Gas_CO2_f3', 'Selec_Top_Gas_H2_f3', 'Selec_Top_Gas_N2_f3',
                            'NE_Uptake_f3','SE_Uptake_f3','NW_Uptake_f3','SW_Uptake_f3','Slag_SiO2_f2','Slag_CaO_f2',
                            'Slag_MgO_f2','SNORT_VALVE_POSITION_f3','TOP_PRESS_f3','HOT_BLAST_PRESSURE_f3',
                            'HOT_METAL_TEMP_f3','cokerate_f4']

        self.l_map_dates = ['SI_f1',
                            'Date_Map',
                            'HOST_BLAST_MOISTURE_f3','HOT_BLAST_TMP_NS_f3','NAT_GAS _INJECTION_f3','WINDRATE_f3',
                            'HIGH_PURITY_OXYGEN_f3','COAL_FLOW_f3','Cast_Avg_Mn_f2','Slag_Fe_f2',
                            'CNUM',
                            'Selec_Top_Gas_CO_f3','Selec_Top_Gas_CO2_f3','Selec_Top_Gas_H2_f3','Selec_Top_Gas_N2_f3',
                            'NE_Uptake_f3','SE_Uptake_f3','NW_Uptake_f3','SW_Uptake_f3','Slag_SiO2_f2','Slag_CaO_f2',
                            'Slag_MgO_f2','SNORT_VALVE_POSITION_f3','TOP_PRESS_f3','HOT_BLAST_PRESSURE_f3',
                            'HOT_METAL_TEMP_f3','cokerate_f4']

        self.l_delta     = ['delta_SI',
                            'SI_f1',          
                            'HOST_BLAST_MOISTURE_f3','HOT_BLAST_TMP_NS_f3','NAT_GAS _INJECTION_f3','WINDRATE_f3',
                            'HIGH_PURITY_OXYGEN_f3','COAL_FLOW_f3','Cast_Avg_Mn_f2','Slag_Fe_f2',
                            'Selec_Top_Gas_CO_f3','Selec_Top_Gas_CO2_f3','Selec_Top_Gas_H2_f3','Selec_Top_Gas_N2_f3',
                            'NE_Uptake_f3','SE_Uptake_f3','NW_Uptake_f3','SW_Uptake_f3','Slag_SiO2_f2','Slag_CaO_f2',
                            'Slag_MgO_f2','SNORT_VALVE_POSITION_f3','TOP_PRESS_f3','HOT_BLAST_PRESSURE_f3',
                            'HOT_METAL_TEMP_f3','cokerate_f4']
        

    def read_data(self, file1 ):
        self.dataFile           = file1
        self.for_RNN_data_CIVS  = pd.read_csv( self.dataFile )
        self.cols_list_DF = self.for_RNN_data_CIVS.columns.values.tolist()

    
    def read_df_Dates_Map(  self, file_name ):

        self.filename_dates_map = file_name
        self.df_Dates_Map  = pd.read_csv( self.filename_dates_map )

    def calculateDeltas(self):
        self.delta_for_RNN_data_CIVS.insert(loc = 0, column = 'delta_SI', value = 0 )
        for index, row in self.delta_for_RNN_data_CIVS.iterrows():
            if index > 1:
                a = self.delta_for_RNN_data_CIVS.at[ index  ,'SI_f1']
                b = self.delta_for_RNN_data_CIVS.at[ index-1,'SI_f1']
                self.delta_for_RNN_data_CIVS.at[index,'delta_SI'] = a - b 

    def calculateMovingAverage(self):
        self.delta_for_RNN_data_CIVS.insert(loc = 1, column = 'mov_avg_SI', value = 0 )
        range_to_mean = 10
        for i in range( 1, self.delta_for_RNN_data_CIVS.shape[0] ):
            index = self.delta_for_RNN_data_CIVS.shape[0] - i
            if index <= 10:
                ## range_to_mean = index -1
                break
            list_to_mean = []
            for j in range( range_to_mean ):
                list_to_mean.append(   self.delta_for_RNN_data_CIVS.at[ index-j , 'delta_SI' ]    )
            the_mean = np.array( list_to_mean )
            self.delta_for_RNN_data_CIVS.at[index,'mov_avg_SI']= np.mean(the_mean) 
     

    def printName(self):
        print( self.MyName  )

    def add_dates_after_GPT_is_trained(self, tsGPT_obj ):
        sliced_chunk_DATES_MAP    = self.df_Dates_Map[ tsGPT_obj.index_to_slice : tsGPT_obj.index_to_slice + 250]['Date_Map']
        sliced_chunk_DATES_MAP_np = sliced_chunk_DATES_MAP.to_numpy()
        sliced_chunk_DATES_MAP_np = np.expand_dims(sliced_chunk_DATES_MAP_np, axis=1)
        excel_matrix_pd           = pd.DataFrame( tsGPT_obj.excel_matrix )
        excel_matrix_pd.to_csv("for_excel_7_window.csv")
        line = 'id, delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,'
        line = line + 'delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,'
        line = line + 'delta_real,delta_pred,DrZ_real,DrZ_pred,None,None'

        with open("for_excel_7_window.csv", 'r+') as file: 
            file_data = file.read() 
            file.seek(0, 0) 
            file.write(line + '\n' + file_data)
        file.close()


    def data_plot_all_columns( self ):

        self.for_RNN_data_CIVS.plot( kind='line',  subplots=True, figsize=(20,80),
                                     sharex=False,  sharey=False, legend=True        )













