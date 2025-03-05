import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_class import CrossBorderData
import data_loader
import config

save_plot = True
model_name = config.MODEL_NAME

pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions', f"{model_name}_{data_loader.DATASET_NAME}_{data_loader.DOMAIN}.csv")
fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/plots/pred', f"{model_name}_{data_loader.DATASET_NAME}_{data_loader.DOMAIN}.png")

predictions_df = pd.read_csv(pred_path)
predictions_df = pd.DataFrame({
    "timestamp": pd.to_datetime(predictions_df["timestamp"]),
    "predicted_capacity": predictions_df["predicted_capacity"]
})

test_dataset = CrossBorderData(data_loader.COUNTRY1, data_loader.COUNTRY2, data_loader.DOMAIN, data_loader.DATASET_NAME, load_from_file=True)
actual_df = pd.DataFrame({
    "timestamp": pd.to_datetime(test_dataset.timestamp),
    "actual_capacity": test_dataset.y.flatten()
})

comparison_df = actual_df.merge(predictions_df, on="timestamp")

print(actual_df.head())
print(predictions_df.head())
print(comparison_df.head())

plt.figure(figsize=(12, 6))
plt.plot(comparison_df["timestamp"], comparison_df["actual_capacity"], label="Actual Capacity", color="blue")
plt.plot(comparison_df["timestamp"], comparison_df["predicted_capacity"], label="Predicted Capacity", color="red", linestyle="dashed")
plt.xlabel("Timestamp")
plt.ylabel("Cross-Border Capacity")
plt.title("Predicted vs. Actual Cross-Border Transmission Capacity")
plt.legend()
plt.grid()
if save_plot:
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("Figure saved at:", f"{fig_path}")
plt.show()
