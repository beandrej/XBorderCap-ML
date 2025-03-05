


# -------  MODELING PARAMETERS ---------
MODEL_TYPE = 0
VALID_MODELS = ['linreg',
                'reg'  
                'nn',       
                'lstm']     
MODEL_NAME = VALID_MODELS[MODEL_TYPE]

# LSTM
HIDDEN_DIM = 128 
NUM_LAYERS = 3  
DROPOUT_LSTM = 0.3  
# NN
DROPOUT_NN = 0.3


# -------  TRAINING PARAMETERS ---------

TRAIN_SPLIT = 0.8 
BATCH_SIZE = 128  
EPOCHS = 75
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.005

