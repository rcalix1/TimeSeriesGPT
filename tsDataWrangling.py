
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
        self.MyName             = 'dataWrangling'
        self.dataFile           = ''
        self.filename_dates_map = ''
        self.for_RNN_data_CIVS  = ''
        self.df_Dates_Map       = ''
        self.cols_list_DF       = None

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
        

    def read_data(self):

        self.for_RNN_data_CIVS  = pd.read_csv( self.dataFile )
        
        self.cols_list_DF = self.for_RNN_data_CIVS.columns.values.tolist()

    
    def read_df_Dates_Map(  self, file_name ):

        self.filename_dates_map = file_name
        self.df_Dates_Map  = pd.read_csv( self.filename_dates_map )


    def printName(self):
        print( self.MyName  )


    def data_plot_all_columns( self ):

        self.for_RNN_data_CIVS.plot( kind='line',  subplots=True, figsize=(20,80),
                                     sharex=False,  sharey=False, legend=True        )













