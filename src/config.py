import os
import sys; sys.dont_write_bytecode = True
this_module = sys.modules[__name__]
import warnings
warnings.filterwarnings("ignore", message=".*Cannot set number of intraop threads.*")
import torch
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler



"""
********************************* MAIN ******************************************
"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

DATASETS = ["BL_FBMC_NORM", "BL_NTC_NORM", "FX_NTC_NORM", "FX_FBMC_NORM"]   
MODELS = ["TCNHybrid"] 

KEEP_BORDER = False
BORDER_TO_KEEP = 'GER_FRA'

TRAIN = True
PREDICT = True
WRITE_METRICS = True
WRITE_PREDICTIONS = True  
OVERWRITE_PREDICTIONS = True 

ENABLE_BACKTEST = False
"""
******************************** TRAINING *********************************************
"""

REG_CRITERIA = nn.MSELoss
CLS_CRITERIA = nn.CrossEntropyLoss
SCALER = MinMaxScaler
OPTIMIZER = torch.optim.AdamW

EPOCHS = 50                 
TRAIN_SPLIT = 0.95              # [%] Portion of FULL SET used for training, rest is test
VALID_SPLIT = 0.20              # [%] Portion of TRAINING SET used for validation
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 3e-4
DROPOUT = 0.4
LEAKY_RELU = 0.05

SHARED_HIDDEN = 256
CLS_HIDDEN = 256
REG_HIDDEN = 128

NUM_WORKERS = 8
SHUFFLE_TRAIN = True

USE_PCA = True                  # True = Use PCA
PCA_COMP = 64                   # PCA output dimension
USE_RF = False                  # Aborts training -> ONLY RF for Dataset

DILATION = [2, 8, 12, 24]       # For TCN if dilation is used
KERNEL_SIZE = 8                 # Kernel size TCN
SEQ_LEN = 48                    # Sequence length for LSTM, TCN (3D-Input models)
           
MIN_EPOCHS_MODEL_SAVE = 10      # model.pth only saves after min. epochs
SAVE_PLOTS = True
SHOW_PLOTS = False

"""
*************************************************************************************
"""

        

FBMC_BORDERS = [
    "AUS_CZE", "CZE_AUS", "AUS_GER", "GER_AUS", "BEL_FRA", "FRA_BEL", "BEL_GER", "GER_BEL", 
    "BEL_NET", "NET_BEL", "CZE_GER", "CZE_POL", "POL_CZE", "GER_NET", "NET_GER", "GER_POL", 
    "POL_GER", "GER_FRA", "FRA_GER"
]

NTC_BORDERS = [
    "AUS_ITA", "ITA_AUS", "AUS_SWI", "SWI_AUS", "BEL_GBR", "GBR_BEL", "SVN_ITA", "ITA_SVN",
    "DK1_GER", "GER_DK1", "DK1_NET", "NET_DK1", "DK2_GER", "GER_DK2", "ESP_FRA", "FRA_ESP",
    "ESP_POR", "POR_ESP", "FRA_GBR", "GBR_FRA", "FRA_ITA", "ITA_FRA", "FRA_SWI", "SWI_FRA",
    "GBR_NET", "NET_GBR", "GER_SWI", "SWI_GER", "ITA_SWI", "SWI_ITA", "NET_NO2", "NO2_NET"
]

REG_COLS = [
    'SWI_ITA', 'FRA_ITA', 'SVN_ITA', 'POR_ESP', 'AUS_ITA', 'ESP_POR', 'ESP_FRA', 'GER_DK1', 
    'DK1_GER', 'FRA_ESP', "AUS_CZE", "CZE_AUS", "AUS_GER", "GER_AUS", "BEL_FRA", "FRA_BEL", 
    "BEL_GER", "GER_BEL", "BEL_NET", "NET_BEL", "CZE_GER", "CZE_POL", "POL_CZE", "GER_NET", 
    "NET_GER", "GER_POL", "POL_GER", "GER_FRA", "FRA_GER"
]

CLS_COLS = [
    'GBR_BEL', 'BEL_GBR', 'ITA_SVN', 'NET_DK1', 'ITA_FRA', 'ITA_SWI', 'GER_SWI', 'ITA_AUS', 
    'FRA_SWI', 'SWI_GER', 'AUS_SWI', 'GBR_NET', 'NET_GBR', 'SWI_FRA', 'GBR_FRA', 'SWI_AUS', 
    'FRA_GBR', 'NO2_NET', 'GER_DK2', 'DK1_NET', 'NET_NO2', 'DK2_GER'
]

ALL_BORDERS = [
    'SWI_ITA', 'FRA_ITA', 'SVN_ITA', 'POR_ESP', 'AUS_ITA', 'ESP_POR', 'ESP_FRA', 'GER_DK1', 
    'DK1_GER', 'FRA_ESP', 'AUS_CZE', 'CZE_AUS', 'AUS_GER', 'GER_AUS', 'BEL_FRA', 'FRA_BEL', 
    'BEL_GER', 'GER_BEL', 'BEL_NET', 'NET_BEL', 'CZE_GER', 'CZE_POL', 'POL_CZE', 'GER_NET', 
    'NET_GER', 'GER_POL', 'POL_GER', 'GER_FRA', 'FRA_GER', 'GBR_BEL', 'BEL_GBR', 'ITA_SVN', 
    'NET_DK1', 'ITA_FRA', 'ITA_SWI', 'GER_SWI', 'ITA_AUS', 'FRA_SWI', 'SWI_GER', 'AUS_SWI', 
    'GBR_NET', 'NET_GBR', 'SWI_FRA', 'GBR_FRA', 'SWI_AUS', 'FRA_GBR', 'NO2_NET', 'GER_DK2', 
    'DK1_NET', 'NET_NO2', 'DK2_GER'
]

def borderCheck(dataset):
    if dataset.split('_')[1] == 'FBMC':
        return FBMC_BORDERS
    elif dataset.split('_')[1] == 'NTC':
        return NTC_BORDERS

def run():
    print(f"\n ------------ TRAINING PIPELINE STARTED ------------ ")

    assert set(ALL_BORDERS) == set(REG_COLS) | set(CLS_COLS), "\n ------- Regression and Cls borders DO NOT match ALL BORDERS! ------- \n"
    assert set(ALL_BORDERS) == set(FBMC_BORDERS) | set(NTC_BORDERS), "\n ------- FBMC and NTC borders DO NOT match with ALL BORDERS! ------- \n"

    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    from training.train_Hybrid import main as train_main

    for model_name in MODELS:
        for dataset in DATASETS:
            for border in borderCheck(dataset):
                print(f"\n ----------------------------- {model_name} on {dataset} | Target: {border} | Border Type: {dataset.split('_')[1]} ----------------------------- ")
                train_main(dataset, model_name, border)



if __name__ == '__main__':
    run()