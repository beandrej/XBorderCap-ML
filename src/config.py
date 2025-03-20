
CURRENT_DF = "BASELINE_MAXBEX"

# -------  MODELING PARAMETERS ---------
MODEL_TYPE = 1
VALID_MODELS = ['linreg',
                'reg',  
                'nn',       
                'lstm']     
MODEL_NAME = VALID_MODELS[MODEL_TYPE]

# LSTM
HIDDEN_DIM = 128 
NUM_LAYERS = 3  
DROPOUT_LSTM = 0.3  
# NN
DROPOUT_NN = 0.0


# -------  TRAINING PARAMETERS ---------

TRAIN_SPLIT = 0.8 
VALID_SPLIT = 0.1
BATCH_SIZE = 512
EPOCHS = 100
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 0.01

