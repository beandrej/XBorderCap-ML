import os
import sys; sys.dont_write_bytecode = True
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler



"""
********************************* MAIN ******************************************
"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

DATASETS = ["FX_NTC_NORM", "BL_NTC_NORM", "BL_FBMC_NORM", "FX_FBMC_NORM"]   
MODELS = ["Hybrid", "LSTM", "Net", "BaseModel"] 

DO_TRAIN = False
DO_PREDICT = True
DO_PLOT = False

"""
******************************** TRAINING *********************************************
"""

REG_CRITERIA = nn.MSELoss
CLS_CRITERIA = nn.CrossEntropyLoss
SCALER = MinMaxScaler
OPTIMIZER = torch.optim.Adam

EPOCHS = 40                     
TRAIN_SPLIT = 0.95              # [%] Portion of FULL SET used for training, rest is test
VALID_SPLIT = 0.20              # [%] Portion of TRAINING SET used for validation
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 3e-4
DROPOUT = 0.4

SEQ_LEN = 8                     # Shot-term memory length LSTM
USE_PCA = True                  # True = Use PCA
PCA_COMP = 64                   # PCA output dimension
ROLLING_HOURS = []              # Every feature col gets averages for the last X hours -> each hour entry doubles the dataset
HIDDEN_DIM = 128                # Nr of nodes for each layer in each Model
UNIQUE_VAL_TRSH = 200           # Hybrid model threshhold Regression <-> Classification
MIN_EPOCHS_MODEL_SAVE = 15      # Model.pth only saves after min. epochs

"""
*************************************************************************************
"""

USE_RF = False                  # Create RF Analysis and plot
PREDICT_ALL_BORDERS = False     # True = Predict all borders simoultaneosly
USE_ROLLING_VAL = False         # True = Pick VALID_SPLITS most similar mean/var score to training
CLASSIFY_WHOLE_DATASET = False  # True = UNIQUE_VAL_TRSH is infinite

SAVE_PLOTS = True               
MAKE_PLOTS = True
SHOW_PLOTS = False
PLOT_LAG_CORR = False
PLOT_BORDER_SPLIT = False       


PREDICT_ON_FULL_DATA = False

FBMC_BORDERS = ["AUS_CZE", "CZE_AUS", "AUS_GER", "GER_AUS","BEL_FRA","FRA_BEL","BEL_GER","GER_BEL","BEL_NET","NET_BEL","CZE_GER",
                "CZE_POL","POL_CZE","GER_NET","NET_GER","GER_POL","POL_GER","GER_FRA","FRA_GER"]

NTC_BORDERS = [
"AUS_ITA","ITA_AUS","AUS_SWI","SWI_AUS","BEL_GBR","GBR_BEL","SVN_ITA","ITA_SVN",
"DK1_GER","GER_DK1","DK1_NET","NET_DK1","DK2_GER","GER_DK2","ESP_FRA","FRA_ESP",
"ESP_POR","POR_ESP","FRA_GBR","GBR_FRA","FRA_ITA","ITA_FRA","FRA_SWI","SWI_FRA",
"GBR_NET","NET_GBR","GER_SWI","SWI_GER","ITA_SWI","SWI_ITA","NET_NO2","NO2_NET"
]