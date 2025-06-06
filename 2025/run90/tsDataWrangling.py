## Author: Ricardo A. Calix, Ph.D.
## Data Wrangling Module

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
from mlxtend.plotting import heatmap
import mlxtend
import matplotlib 



plt.rcParams["figure.figsize"] = [15, 5]
pd.options.display.max_columns = None
## pd.set_option('display.max_columns', None)
pd.options.display.max_rows    = 30




############################################################

class tsDataWrangling:
    
    def __init__(self):
        self.MyName                             = 'dataWrangling'
        self.dataFile                           = ''
        self.filename_dates_map                 = ''
        self.delta_for_RNN_data_CIVS            = None
        self.SI_lookup_for_RNN_data_CIVS        = None
        self.df_Dates_Map                       = None
        self.cols_list_DF                       = None
        #####################
        self.coke_rate_only_f4_pd               = None
        self.coke_rate_only_f4_pd_265k          = None
        self.for_RNN_data_CIVS                  = None
        self.MinByMin_137MB_data_CIVS           = None
        self.all_columns_in_DF                  = None
        self.selected_cols_for_RNN_data_CIVS_df = None
        self.cols_rotation_DF                   = None
        self.dict_map_cnum_dates                = {}
        self.FOUR_files_merged_data_rc          = None
        self.res                                = None
        self.df_res                             = None
        self.df_res_shifted                     = None 
        self.dates_df_res_shifted               = None
        

        self.l_delta     = ['delta_SI',
                            'SI_f1',          
                            'HOST_BLAST_MOISTURE_f3','HOT_BLAST_TMP_NS_f3','NAT_GAS _INJECTION_f3','WINDRATE_f3',
                            'HIGH_PURITY_OXYGEN_f3','COAL_FLOW_f3','Cast_Avg_Mn_f2','Slag_Fe_f2',
                            'Selec_Top_Gas_CO_f3','Selec_Top_Gas_CO2_f3','Selec_Top_Gas_H2_f3','Selec_Top_Gas_N2_f3',
                            'NE_Uptake_f3','SE_Uptake_f3','NW_Uptake_f3','SW_Uptake_f3','Slag_SiO2_f2','Slag_CaO_f2',
                            'Slag_MgO_f2','SNORT_VALVE_POSITION_f3','TOP_PRESS_f3','HOT_BLAST_PRESSURE_f3',
                            'HOT_METAL_TEMP_f3','cokerate_f4']

        

        self.selected_columns_RNN = ['SI_f1','Timestamp_f1', 'MM_Timestamp_f1',  'HOST_BLAST_MOISTURE_f3', 'HOT_BLAST_TMP_NS_f3', 
                            'NAT_GAS _INJECTION_f3',  'WINDRATE_f3',  'HIGH_PURITY_OXYGEN_f3', 'COAL_FLOW_f3', 'cokerate_f4', 
                            'Cast_Avg_Mn_f2', 'Slag_Fe_f2',  'date_f1', 'CNUM', 'LNUM_f1',
                            'Selec_Top_Gas_CO_f3',  'Selec_Top_Gas_CO2_f3', 'Selec_Top_Gas_H2_f3', 'Selec_Top_Gas_N2_f3',
                            'NE_Uptake_f3', 'SE_Uptake_f3', 'NW_Uptake_f3', 'SW_Uptake_f3',  
                            'Slag_SiO2_f2', 'Slag_CaO_f2', 'Slag_MgO_f2', 'SNORT_VALVE_POSITION_f3', 'TOP_PRESS_f3', 'HOT_BLAST_PRESSURE_f3',
                            'Taphole_f2','HOT_METAL_TEMP_f3']
    

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
                            'Taphole_f2', 'HOT_METAL_TEMP_f3','cokerate_f4']

        
        self.l_mod_step1 = ['THEORETICAL_TONNAGE_f3','Coke_Rate_f3','HOT_BLAST_TMP_NS_f3','WINDRATE_f3','SNORT_VALVE_POSITION_f3',
                            'COLD_BLAST_MOISTURE_f3','HOST_BLAST_MOISTURE_f3','HIGH_PURITY_OXYGEN_f3','COAL_FLOW_f3',
                            'NAT_GAS _INJECTION_f3', 'HOT_BLAST_PRESSURE_f3','TOP_PRESS_f3','TOP_TEMP_HIGHEST_f3',
                            'Top_Gas_ETACO_f3 ', 'Selec_Top_Gas_CO_f3','Selec_Top_Gas_CO2_f3','Selec_Top_Gas_H2_f3',
                            'Selec_Top_Gas_N2_f3','SILICON_f3','SULFUR_f3','MANGANESE_f3','PHOSPHOROUS_f3','HOT_METAL_TEMP_f3',
                            'Slag_CaO_day_avg_f3','Slag_MgO_day_avg_f3','Slag_SiO2_f3','Slag_Al2O3_day_avg_f3',
                            '#14 WEST STOCKROD LEVEL','#14 EAST STOCKROD LEVEL','#14 RADAR ROD NORTH LEVEL',
                            '#14 RADAR ROD SOUTH LEVEL','Charges_per_hour_f3','BF14 material 1 weight','BF14 extra coke weight',
                            'BF14 material 2 weight','BF14 material 2 weight.1','BF14 material 3 weight','BF14 material 3 weight.1',
                            'BF14 material 4 weight','BF14 material 4 weight.1','BF14 material 5 weight','BF14 material 5 weight.1',
                            'BF14 material 6 weight','BF14 material 6 weight.1','BF14 material 7 weight','BF14 material 7 weight.1',
                            'BF14 material 8 weight','BF14 material 8 weight.1','BF14 material 9 weight','BF14 material 9 weight.1',
                            'BF14 material 10 weight','BF14 material 10 weight.1','NE_Uptake_f3','SE_Uptake_f3',
                            'NW_Uptake_f3','SW_Uptake_f3']
        
        
        self.selected_columns_RNN_no_dates = [ 'SI_f1', 'HOST_BLAST_MOISTURE_f3', 'HOT_BLAST_TMP_NS_f3', 
                            'NAT_GAS _INJECTION_f3',  'WINDRATE_f3',  'HIGH_PURITY_OXYGEN_f3', 'COAL_FLOW_f3', 'cokerate_f4', 
                            'Cast_Avg_Mn_f2', 'Slag_Fe_f2',  'date_f1', 'CNUM', 'LNUM_f1',
                            'Selec_Top_Gas_CO_f3',  'Selec_Top_Gas_CO2_f3', 'Selec_Top_Gas_H2_f3', 'Selec_Top_Gas_N2_f3',
                            'NE_Uptake_f3', 'SE_Uptake_f3', 'NW_Uptake_f3', 'SW_Uptake_f3',  
                            'Slag_SiO2_f2', 'Slag_CaO_f2', 'Slag_MgO_f2', 'SNORT_VALVE_POSITION_f3', 'TOP_PRESS_f3', 'HOT_BLAST_PRESSURE_f3',
                            'Taphole_f2','HOT_METAL_TEMP_f3']

        

        self.cols_to_use = [ 'SI_f1', 'HOST_BLAST_MOISTURE_f3', 'HOT_BLAST_TMP_NS_f3', 
                            'NAT_GAS _INJECTION_f3',  'WINDRATE_f3',  'HIGH_PURITY_OXYGEN_f3', 'COAL_FLOW_f3', 
                            'Cast_Avg_Mn_f2', 'Slag_Fe_f2',  'date_f1', 'CNUM', 'LNUM_f1',
                            'Selec_Top_Gas_CO_f3',  'Selec_Top_Gas_CO2_f3', 'Selec_Top_Gas_H2_f3', 'Selec_Top_Gas_N2_f3',
                            'NE_Uptake_f3', 'SE_Uptake_f3', 'NW_Uptake_f3', 'SW_Uptake_f3',  
                            'Slag_SiO2_f2', 'Slag_CaO_f2', 'Slag_MgO_f2', 'SNORT_VALVE_POSITION_f3', 'TOP_PRESS_f3', 'HOT_BLAST_PRESSURE_f3',
                            'Taphole_f2','HOT_METAL_TEMP_f3']

        
        self.cols_to_use_backup = ['SI_f1','HOST_BLAST_MOISTURE_f3','HOT_BLAST_TMP_NS_f3',
                            'NAT_GAS _INJECTION_f3','WINDRATE_f3','HIGH_PURITY_OXYGEN_f3','COAL_FLOW_f3',
                            'Cast_Avg_Mn_f2','Slag_Fe_f2',
                            'Selec_Top_Gas_CO_f3','Selec_Top_Gas_CO2_f3','Selec_Top_Gas_H2_f3','Selec_Top_Gas_N2_f3',
                            'NE_Uptake_f3','SE_Uptake_f3','NW_Uptake_f3','SW_Uptake_f3',
                            'Slag_SiO2_f2','Slag_CaO_f2','Slag_MgO_f2','SNORT_VALVE_POSITION_f3','TOP_PRESS_f3','HOT_BLAST_PRESSURE_f3',
                            'HOT_METAL_TEMP_f3']

        

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
                ## print(float( a )  - float( b ) )
                self.delta_for_RNN_data_CIVS.at[ index, 'delta_SI'] = float( a )  - float( b ) 
                self.delta_for_RNN_data_CIVS.loc[index, 'delta_SI'] = float( a )  - float( b )  
                ## print( self.delta_for_RNN_data_CIVS.at[index,'delta_SI']  )
                ## input()

    

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
            self.delta_for_RNN_data_CIVS.at[ index,'mov_avg_SI'] = np.mean(the_mean) 
            self.delta_for_RNN_data_CIVS.loc[index,'mov_avg_SI'] = np.mean(the_mean)
     

    
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
        

    def step1_iterate_func( self ):
        self.for_RNN_data_CIVS['Timestamp_f1'] = pd.to_datetime(self.for_RNN_data_CIVS['Timestamp_f1'], unit='D', origin='1899-12-30') 
        self.for_RNN_data_CIVS['Timestamp_f1'] = pd.to_datetime(self.for_RNN_data_CIVS['Timestamp_f1']).round('min') 
        self.MinByMin_137MB_data_CIVS["MM_Timestamp_f1"] = self.MinByMin_137MB_data_CIVS["MM_Timestamp_f1"].astype("datetime64[ns]")
        list_dates_6000f = self.for_RNN_data_CIVS['Timestamp_f1'].tolist()
        self.MinByMin_137MB_data_CIVS.insert(loc = 0, column = 'Timestamp_f1', value = 'abc')
        i_6000 = 0
        merge_date_key = list_dates_6000f[ i_6000 ] 
        for index, row in self.MinByMin_137MB_data_CIVS.iterrows():
            if row["MM_Timestamp_f1"] <= merge_date_key and row['Timestamp_f1'] == 'abc':
                self.MinByMin_137MB_data_CIVS.at[index, 'Timestamp_f1'] = merge_date_key
            else:
                i_6000  = i_6000 + 1
                if i_6000 >= len( list_dates_6000f ):
                    break
                merge_date_key = list_dates_6000f[ i_6000 ] 
                self.MinByMin_137MB_data_CIVS.at[index, 'Timestamp_f1'] = merge_date_key
            if index % 50000 == 0:
                print( index )
        self.MinByMin_137MB_data_CIVS = self.MinByMin_137MB_data_CIVS.drop( 
                        self.MinByMin_137MB_data_CIVS[ self.MinByMin_137MB_data_CIVS[ 'Timestamp_f1'] == 'abc' ].index
        )
        ## self.MinByMin_137MB_data_CIVS.to_csv('input/step2/min_by_bin_mapped_dates_264917_rcalix.csv')
        

    
    def step1_iterate_coke_rate( self ):
        ## print(self.coke_rate_only_f4_pd)
        aaa = np.repeat( self.coke_rate_only_f4_pd.values, 60, axis=0 )
        rc = 0
        for i in range( len(aaa) ):
            temp = aaa[i, 0].split(":")
            if rc < 10:
                rc_str = "0" + str(rc)
            else:
                rc_str = str( rc )
            temp[1] = rc_str 
            aaa[i,0] = temp[0] + ":" + temp[1] + ":" + temp[2]
            rc = rc + 1
            if rc == 60:
                rc = 0
        newdf         = pd.DataFrame( aaa )
        newdf.columns = self.coke_rate_only_f4_pd.columns
        self.coke_rate_only_f4_pd_265k = newdf
        self.coke_rate_only_f4_pd_265k["MM_Timestamp_f1"] = self.coke_rate_only_f4_pd_265k["MM_Timestamp_f1"].astype("datetime64[ns]")
        ## print( self.coke_rate_only_f4_pd_265k )
        

    
    def step1_the_merge(self):
        l1_min_by_min = self.MinByMin_137MB_data_CIVS.columns.values.tolist() 
        self.MinByMin_137MB_data_CIVS['Timestamp_f1']  = self.MinByMin_137MB_data_CIVS['Timestamp_f1'].astype(str)
        self.MinByMin_137MB_data_CIVS[self.l_mod_step1]= self.MinByMin_137MB_data_CIVS[self.l_mod_step1].apply(pd.to_numeric,errors='coerce')
        self.for_RNN_data_CIVS['Timestamp_f1']         = self.for_RNN_data_CIVS['Timestamp_f1'].astype(str)
        merged_data_rc = pd.merge( self.MinByMin_137MB_data_CIVS , self.for_RNN_data_CIVS, on='Timestamp_f1', how='left')
        merged_data_rc['MM_Timestamp_f1'] = merged_data_rc['MM_Timestamp_f1'].astype(str)
        self.coke_rate_only_f4_pd_265k['MM_Timestamp_f1'] = self.coke_rate_only_f4_pd_265k['MM_Timestamp_f1'].astype(str)
        self.FOUR_files_merged_data_rc = pd.merge( merged_data_rc , self.coke_rate_only_f4_pd_265k, on='MM_Timestamp_f1', how='left')
        self.FOUR_files_merged_data_rc.to_csv('input/step2/FINAL_RNN_mapped_dates_266096_rcalix_THIS_ONE.csv')
        
        

    def step1_wrangle(self):
        self.coke_rate_only_f4_pd     = pd.read_csv('input/step1/Coke_Rate_only_f4.csv'   )
        self.for_RNN_data_CIVS        = pd.read_csv('input/step1/merged_seq_5998_CIVS.csv')
        self.MinByMin_137MB_data_CIVS = pd.read_csv('input/step1/CIVS_137_MB_MinbyMin.csv')
        self.step1_iterate_coke_rate()
        self.step1_iterate_func()
        self.step1_the_merge()
        

    def remove_downtime(self, data_frame):
        data_frame = pd.DataFrame(data_frame)
        data_frame = data_frame.replace(np.nan,0)
        data_frame = data_frame[data_frame['SNORT_VALVE_POSITION_f3'].apply(lambda x:   not any(c.isalpha() for c in str(x)))]
        data_frame = data_frame[data_frame['TOP_PRESS_f3'].apply(lambda x:              not any(c.isalpha() for c in str(x)))]
        data_frame = data_frame[data_frame['HOT_BLAST_PRESSURE_f3'].apply(lambda x:     not any(c.isalpha() for c in str(x)))]
        #data_frame = data_frame.rename(columns = {'Unnamed: 0': 'timestamp'})
        data_frame['SNORT_VALVE_POSITION_f3']   = pd.to_numeric( data_frame['SNORT_VALVE_POSITION_f3'] )
        data_frame['TOP_PRESS_f3']              = pd.to_numeric( data_frame['TOP_PRESS_f3'] )
        data_frame['HOT_BLAST_PRESSURE_f3']     = pd.to_numeric( data_frame['HOT_BLAST_PRESSURE_f3'] )
        data_frame['MM_Timestamp_f1'] = pd.to_datetime( data_frame['MM_Timestamp_f1'] )
        data_frame = data_frame[data_frame['SNORT_VALVE_POSITION_f3'] >= 60]
        data_frame = data_frame[data_frame['TOP_PRESS_f3'] >=4]
        data_frame = data_frame[data_frame['HOT_BLAST_PRESSURE_f3']>=5]
        data = data_frame.copy()
        data['timediff'] = data['MM_Timestamp_f1'].diff().dt.total_seconds() / 60.0
        data = data[['MM_Timestamp_f1', 'SNORT_VALVE_POSITION_f3','HOT_BLAST_PRESSURE_f3','TOP_PRESS_f3','timediff']]
        data['idx'] = data.index
        list_f = []
        list_t = []
        data = data.reset_index()
        for i in range(len(data)-1):
            if data['timediff'][i] > 100:
                list_f.append([(data['idx'][i-1]) - 180,(data['idx'][i])+480] )
                list_t.append([
                    (data['MM_Timestamp_f1'][i-1]) - pd.to_timedelta(6, unit='h' ),
                    (data['MM_Timestamp_f1'][i])   + pd.to_timedelta(24, unit='h') 
                ])
        for i in list_f:
            result = data_frame[i[0]:(i[1]+1)]
            if result.empty:
                continue
            else:
                data_frame = data_frame.drop(data_frame.index[i[0]:(i[1]+1)])
        return data_frame


    def step2_initial_data_viewing(self):
        print( self.for_RNN_data_CIVS.head(5)  )
        print( self.for_RNN_data_CIVS.info(verbose=True) ) 
        self.for_RNN_data_CIVS.plot( figsize=(20,80),  subplots=True   )
        print(  self.for_RNN_data_CIVS  )
        print(  self.for_RNN_data_CIVS['MM_Timestamp_f1'][1000:1400]    )

    
    def step2_interpolate_and_check_missing_values(self):
        self.selected_cols_for_RNN_data_CIVS_df = self.selected_cols_for_RNN_data_CIVS_df.interpolate(
                  method ='linear',
                  limit_direction ='forward'
        )
        
        ## null_count = self.selected_cols_for_RNN_data_CIVS_df.isnull().sum().sum()
        ## print('Number of null values:', null_count)
        ## xxkk = sum( map(any, self.selected_cols_for_RNN_data_CIVS_df.isnull()   ))
        ## print( xxkk )
        ## df = self.selected_cols_for_RNN_data_CIVS_df 
        # TOTAL number of missing values:
        ## print( df.isna().sum().sum() )
        # number of ROWS with at least one missing value:
        ## print( (df.isna().sum(axis=1) > 0).sum() )
        # number of COLUMNS with at least one missing value:
        ## print( (df.isna().sum(axis=0) > 0).sum() )
        ## print(  df.isna().any(axis=1).sum() )
        # check for NaN values in each column
        ## print(df.isnull().any())

    
    def step2_temp_plots(self):
        self.selected_cols_for_RNN_data_CIVS_df.plot( figsize=(20,80),  subplots=True   )
        self.selected_cols_for_RNN_data_CIVS_df.hist(column='CNUM')


    def step2_create_dictionary_of_dates(self):
        self.dict_map_cnum_dates = {}
        cnum_old = 0
        for index, row in self.selected_cols_for_RNN_data_CIVS_df.iterrows():
            cnum_new = self.selected_cols_for_RNN_data_CIVS_df.at[index, 'CNUM']
            if cnum_new != cnum_old:
                self.dict_map_cnum_dates[ cnum_new ] = self.selected_cols_for_RNN_data_CIVS_df.at[index, 'Timestamp_f1']
                cnum_old                             = cnum_new 

    
    def step2_plot_correlation_matrix( self, res ):
        print(  res.info()  )
        print(  res.shape   )
        res.plot( figsize=(20,80),  subplots=True   )
        headers_list = res.columns.values.tolist()
        print(  headers_list  )
        cm = np.corrcoef(  res[ headers_list ].values.T  )
        hm = heatmap(cm, row_names= headers_list, column_names=headers_list, figsize=[20,10])
        plt.show()
        print(  res.info()  )

    
    def step2_simpler_plot_correlation_matrix( self, res ):
        print(  res.info()  )
        print(  res.shape   )
        headers_list = res.columns.values.tolist()
        print(  headers_list  )
        cm = np.corrcoef(  res[ headers_list ].values.T  )
        hm = heatmap(cm, row_names= headers_list, column_names=headers_list, figsize=[20,10])
        plt.show()



    def step2_mean_on_CNUM_and_remove_some_peaks(self):
        res = self.selected_cols_for_RNN_data_CIVS_df.groupby('CNUM')[  self.selected_columns_RNN_no_dates   ].mean()
        res = res.reset_index(drop=True)
        ## res.plot( figsize=(20,80),  subplots=True   )
        res = res[ res['Slag_CaO_f2'] >= 25]
        res = res[ res['Cast_Avg_Mn_f2'] >= 0.2 ]
        res = res[ res['Slag_Fe_f2'] < 1.5 ]
        res = res[ res['COAL_FLOW_f3'] < 2 ]
        res = res[ res['HOT_BLAST_PRESSURE_f3'] > 25 ]
        res = res[ res['TOP_PRESS_f3'] > 14 ]
        res = res.reset_index(drop=True)
        ## self.step2_plot_correlation_matrix( res )
        return res
        
        
    def step2_Previous_1_Cast_Processing(self,  res):
        self.cols_rotation_DF = res.columns.values.tolist()
        idx = res.index[ : -1]
        y_1_cast = res.iloc[ 1:   ,  0 ].values
        x_1_cast = res.iloc[  :-1 , 1: ].values
        df_xs  = pd.DataFrame(x_1_cast , columns=  self.cols_rotation_DF[1:],    index=idx)
        df_y   = pd.DataFrame(y_1_cast , columns=[ self.cols_rotation_DF[0] ],   index=idx)
        df_res = pd.concat( [ df_y, df_xs], axis=1 )     
        ## self.step2_plot_correlation_matrix( df_res )
        return df_res
        

    def step2_get_coke_rate_6_hour(self, res):
        x_coke_6_df   = res['cokerate_f4']
        x_coke_6_hour = x_coke_6_df.iloc[ : -2 ].values
        return x_coke_6_hour
        

    def step2_concat_prevCast_and_cokeRate(self,  df_res, x_coke_6_hour ):
        si_and_1_cast_x       = df_res.iloc[ 1: , :  ].values
        self.cols_rotation_DF = df_res.columns.values.tolist()
        idx                   = df_res.index[ 1: ]
        df_si_x        = pd.DataFrame(si_and_1_cast_x,  columns=self.cols_rotation_DF,        index=idx)
        df_x_coke      = pd.DataFrame(x_coke_6_hour   , columns=['cokerate_f4'],              index=idx)
        df_res_shifted = pd.concat( [ df_si_x, df_x_coke], axis=1 )  
        ## self.step2_plot_correlation_matrix( df_res_shifted )
        return df_res_shifted


    
    def step2_add_dates_to_processed_data(self, df_res_shifted ):
        df_res_shifted.insert(1, 'Date_Map', 'abc')
        for index, row in df_res_shifted.iterrows():
            the_cnum = df_res_shifted.at[index, 'CNUM']
            if self.dict_map_cnum_dates[ the_cnum ]: 
                df_res_shifted.at[index, 'Date_Map'] = self.dict_map_cnum_dates[the_cnum ]
        return df_res_shifted
        

  
    
    def step2_wrangle(self):
        self.for_RNN_data_CIVS = pd.read_csv('input/step2/FINAL_RNN_mapped_dates_266096_rcalix_THIS_ONE.csv')
        ## self.step2_initial_data_viewing()
        ## print( self.for_RNN_data_CIVS.info(verbose=True) )  ## helpful to view features before selection
        self.all_columns_in_DF                  = self.for_RNN_data_CIVS.columns.values.tolist()
        self.selected_cols_for_RNN_data_CIVS_df = self.for_RNN_data_CIVS[  self.selected_columns_RNN  ]
        self.step2_interpolate_and_check_missing_values()
        self.selected_cols_for_RNN_data_CIVS_df = self.remove_downtime(  self.selected_cols_for_RNN_data_CIVS_df  )
        self.selected_cols_for_RNN_data_CIVS_df = self.selected_cols_for_RNN_data_CIVS_df.reset_index()
        ## self.step2_temp_plots()
        self.step2_create_dictionary_of_dates()
        self.res                  = self.step2_mean_on_CNUM_and_remove_some_peaks()
        self.df_res               = self.step2_Previous_1_Cast_Processing( self.res )
        x_coke_6_hour             = self.step2_get_coke_rate_6_hour( self.res )
        self.df_res               = self.df_res[ self.cols_to_use ]    ## removes coke rate, to add with rotation
        self.df_res_shifted       = self.step2_concat_prevCast_and_cokeRate(  self.df_res, x_coke_6_hour )
        df_copy                   = self.df_res_shifted.copy()
        self.dates_df_res_shifted = self.step2_add_dates_to_processed_data(  df_copy  )
        self.df_res_shifted.to_csv(      'input/step3/RNN_time_DELAYS_2000_PerCast_rcalix.csv'              )
        self.dates_df_res_shifted.to_csv('input/step3/dates_RC_CNUM_RNN_time_DELAYS_2000_PerCast_rcalix.csv')

    def step3_breakUP_date_params(self):     ### self.delta_for_RNN_data_CIVS
        
        self.delta_for_RNN_data_CIVS.insert(loc = 9, column = 'year',  value = 0 )
        self.delta_for_RNN_data_CIVS.insert(loc = 9, column = 'month', value = 0 )
        self.delta_for_RNN_data_CIVS.insert(loc = 9, column = 'day',   value = 0 )
        self.delta_for_RNN_data_CIVS.insert(loc = 9, column = 'hour',  value = 0 )
        self.delta_for_RNN_data_CIVS.insert(loc = 9, column = 'min',   value = 0 )
        self.delta_for_RNN_data_CIVS.insert(loc = 9, column = 'sec',   value = 0 )
        
        for index, row in self.delta_for_RNN_data_CIVS.iterrows():
            the_date = self.delta_for_RNN_data_CIVS.at[ index  ,'Date_Map']
            the_date = str(  the_date   )
            temp = the_date.split(" ")
            ## print( temp )
            temp_calendar  = temp[0].split("-")
            calendar_clock = temp[1].split(":")
            self.delta_for_RNN_data_CIVS.at[ index,'year']  = int( temp_calendar[0]  )
            self.delta_for_RNN_data_CIVS.at[ index,'month'] = int( temp_calendar[1]  )
            self.delta_for_RNN_data_CIVS.at[ index,'day']   = int( temp_calendar[2]  )
            self.delta_for_RNN_data_CIVS.at[ index,'hour']  = int( calendar_clock[0] )
            self.delta_for_RNN_data_CIVS.at[ index,'min']   = int( calendar_clock[1] )
            self.delta_for_RNN_data_CIVS.at[ index,'sec']   = int( calendar_clock[2] )


        

    def step3_wrangle(self):
        self.read_df_Dates_Map('input/step3/dates_RC_CNUM_RNN_time_DELAYS_2000_PerCast_rcalix.csv')
        self.df_Dates_Map                 = self.df_Dates_Map[ self.l_map_dates ]    ## checked, should be ok 
        self.for_RNN_data_CIVS            = self.df_Dates_Map 
        self.delta_for_RNN_data_CIVS      = self.for_RNN_data_CIVS.copy()
        self.calculateDeltas()
        self.calculateMovingAverage()
        self.step3_breakUP_date_params()     ### uses self.delta_for_RNN_data_CIVS
        self.for_RNN_data_CIVS            = self.delta_for_RNN_data_CIVS
        self.for_RNN_data_CIVS.to_csv('input/step4/datesBrokenUp_MovingAVG_FINAL_DATA_rcalix.csv')
        
    

    def simulate_better_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a transformed version of the DataFrame with improved learnability,
        without adding columns. Keeps column 2 as the target (e.g., silicon).
        """
        df = df.copy()

        # Create a smooth upward trend in column 2 (target)
        t = np.linspace(0, 1, len(df))
        df.iloc[:, 2] = t + np.random.normal(0, 0.01, size=len(df))  # target: SI

        # Smooth column 5 (simulate better sensor signal)
        df.iloc[:, 5] = df.iloc[:, 5].rolling(window=10, min_periods=1).mean()

        # Replace column 8 with lagged version of target (column 2)
        df.iloc[:, 8] = df.iloc[:, 2].shift(1).bfill()

        # Scale column 10 to be correlated with target
        df.iloc[:, 10] = 3 * df.iloc[:, 2] + np.random.normal(0, 0.01, size=len(df))

        # Optionally smooth another column (e.g., column 12)
        df.iloc[:, 12] = df.iloc[:, 12].ewm(span=8).mean()

        return df


    def simulate_better_dataset_more(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a transformed version of the DataFrame with improved learnability,
        without adding columns. Keeps column 2 as the target (e.g., silicon).
        """
        df = df.copy()

        # Synthetic SI target with a clean trend
        t = np.linspace(0, 1, len(df))
        df.iloc[:, 2] = t + np.random.normal(0, 0.01, size=len(df))  # Column 2 = target (SI)

        # Feature 1: Lagged version of SI
        df.iloc[:, 8] = df.iloc[:, 2].shift(1).bfill()

        # Feature 2: Scaled version of SI
        df.iloc[:, 10] = 3 * df.iloc[:, 2] + np.random.normal(0, 0.01, size=len(df))

        # Feature 3: Smoothed version of an existing column
        df.iloc[:, 12] = df.iloc[:, 5].ewm(span=8).mean()

        # Feature 4: Linear combination of SI and another feature
        df.iloc[:, 15] = 0.4 * df.iloc[:, 2] + 0.6 * df.iloc[:, 10] + np.random.normal(0, 0.01, size=len(df))

        # Feature 5: Noisy irrelevant column (to test model’s ability to ignore junk)
        df.iloc[:, 18] = np.random.normal(0, 1.0, size=len(df))
        
        df.iloc[:, 20] = df.iloc[:, 10] + np.random.normal(0, 0.005, size=len(df))
        
        df.iloc[:, 25] = df.iloc[:, 12] + np.random.normal(0, 0.005, size=len(df))



        return df
    
    


  
    def overwrite_with_sine_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
    
        df = df.copy()
        N = len(df)
        t = np.linspace(0, 4 * np.pi, N)
        sine = np.sin(t)

        def safe_write(col_idx, values):
            if col_idx < df.shape[1]:
                df.iloc[:, col_idx] = values

        # Target column
        safe_write(2, sine)

        # Learnable features
        safe_write(5,  np.roll(sine, 1))                               # lag-1
        safe_write(8,  2 * sine)                                       # scaled
        safe_write(10, np.cos(t))                                      # cosine
        safe_write(12, 0.5 * sine + 0.5 * np.roll(sine, 1))            # avg
        safe_write(15, np.sin(t + np.pi / 4))                          # phase-shifted
        safe_write(18, sine + np.random.normal(0, 0.01, N))            # low noise
        safe_write(20, np.sqrt(np.abs(sine)) * np.sign(sine))         # nonlinear

        # Distractor columns
        safe_write(25, np.random.normal(0, 1.0, N))                    # pure noise
        if df.shape[1] > 30:
            mixed = 0.3 * np.roll(sine, 1) + 0.2 * np.cos(t) + \
                    0.1 * (0.5 * sine + 0.5 * np.roll(sine, 1)) + \
                    0.4 * np.random.randn(N)
            safe_write(30, mixed)

        return df


        

    def data_plot_all_columns( self ):
        self.for_RNN_data_CIVS.plot( kind='line',  subplots=True, figsize=(20,80),
                                     sharex=False,  sharey=False, legend=True        )













