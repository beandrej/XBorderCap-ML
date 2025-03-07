import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_class import *
import data_loader
import config

save_plot = True
model_name = config.MODEL_NAME

pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions', f"{model_name}_{data_loader.DATASET_NAME}_{data_loader.DOMAIN}.csv")
fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/plots/pred', f"{model_name}_{data_loader.DATASET_NAME}_{data_loader.DOMAIN}.png")

full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "MAX_BEX_WITH_FEATURES.csv"), index_col=0)
first_target_idx = full_df.columns.get_loc("AUS_BEL")
Y = full_df.iloc[:, first_target_idx:]

predictions_df = pd.read_csv(pred_path)
comparison_df = Y.merge(predictions_df, on="timestamp", how='inner')

plot_real = comparison_df["AUS_BEL"]
plot_pred = comparison_df["AUS_BEL_pred"]


plt.figure(figsize=(12, 6))
plt.plot(comparison_df["timestamp"], plot_real, label="Actual Capacity", color="blue")
plt.plot(comparison_df["timestamp"], plot_pred, label="Predicted Capacity", color="red", linestyle="dashed")
plt.xlabel("Timestamp")
plt.ylabel("Cross-Border Capacity")
plt.title("Predicted vs. Actual Cross-Border Transmission Capacity")
plt.legend()
plt.grid()
if save_plot:
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("Figure saved at:", f"{fig_path}")
plt.show()
