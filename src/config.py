
# MODEL
VALID_MODELS = ['linreg',   # 0
                'nn',       # 1
                'lstm']     # 2
MODEL_TYPE = 0
MODEL_NAME = VALID_MODELS[MODEL_TYPE]

HIDDEN_DIM = 128 
NUM_LAYERS = 3  
DROPOUT = 0.3  


# TRAINING
TRAIN_SPLIT = 0.9 
BATCH_SIZE = 64  
EPOCHS = 30 
LEARNING_RATE = 0.0005 

