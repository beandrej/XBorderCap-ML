


# -------  MODELING PARAMETERS ---------
MODEL_TYPE = 2
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
DROPOUT_NN = 0.05


# -------  TRAINING PARAMETERS ---------

TRAIN_SPLIT = 0.8 
BATCH_SIZE = 512
EPOCHS = 100
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 0.001

