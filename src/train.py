import pandas as pd
import torch
#from pycaret.regression import *
from model import LinReg

from data_loader import CrossBorderData
from torch.utils.data import DataLoader

# Load training data
train_dataset = CrossBorderData(train=True)
test_dataset = CrossBorderData(train=False)

train_dataset.print_stats()
test_dataset.print_stats()

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}, Total length: {len(test_dataset) + len(train_dataset)}")




# TODO test_set + 1, since last entry and first entry of train & test are same







# # Load preprocessed data
# data = pd.read_csv("processed_data.csv")  # Ensure your preprocessed file exists
# target_column = "cross_border_capacity"  # Define your target variable

# # Initialize PyCaret for regression
# s = setup(data, target=target_column, session_id=123, normalize=True)

# # Compare models
# best_model = compare_models()

# # Print best model
# print(best_model)