import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as pls

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
print(src_path)
sys.path.append(src_path)

# Now import data_loader
from data_loader import CrossBorderData

model_name = "linreg"
pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/predictions', f"{model_name}.csv")

test_dataset = CrossBorderData(train=False) 
actual_df = pd.DataFrame({
    "timestamp": test_dataset.index,
    "actual_capacity": test_dataset.y.numpy().flatten()
})

# TODO maybe change CrossBorderData to pd.Dataframe default -> convert to pytorch tensor in train.py file
# TODO plot some first results
