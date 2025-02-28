import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(src_path)

from data_loader import CrossBorderData
import config

save_plot = False
model_name = config.MODEL_NAME

pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../predictions', f"{model_name}.csv")
predictions_df = pd.read_csv(pred_path)
predictions_df = pd.DataFrame({
    "timestamp": pd.to_datetime(predictions_df["timestamp"]),
    "predicted_capacity": predictions_df["predicted_capacity"]
})

test_dataset = CrossBorderData(train=False).data
actual_df = pd.DataFrame({
    "timestamp": pd.to_datetime(test_dataset.index),
    "actual_capacity": test_dataset["cross_border_capacity"].to_numpy().flatten()
})

comparison_df = actual_df.merge(predictions_df, on="timestamp")
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', f"{model_name}.png")

plt.figure(figsize=(12, 6))
plt.plot(comparison_df["timestamp"], comparison_df["actual_capacity"], label="Actual Capacity", color="blue")
plt.plot(comparison_df["timestamp"], comparison_df["predicted_capacity"], label="Predicted Capacity", color="red", linestyle="dashed")
plt.xlabel("Timestamp")
plt.ylabel("Cross-Border Capacity")
plt.title("Predicted vs. Actual Cross-Border Transmission Capacity")
plt.legend()
plt.grid()
if save_plot:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Figure saved at:", f"{save_path}")
plt.show()
