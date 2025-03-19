import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config
import data_loader

save_plot = True
model_name = config.MODEL_NAME

# Paths
pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results/predictions', f"{model_name}_{data_loader.DATASET_NAME}.csv")
fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results/plots/pred', f"{model_name}_{data_loader.DATASET_NAME}_{data_loader.DOMAIN}.png")

# Load actual Y data
full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '../prep_data', "BASELINE_MAXBEX.csv"), index_col=0)
first_target_idx = full_df.columns.get_loc("AUS_CZE")
Y = full_df.iloc[:, first_target_idx:].reset_index()

# Load predictions
predictions_df = pd.read_csv(pred_path)

# Merge on timestamp
comparison_df = pd.merge(Y, predictions_df, on="timestamp", how='inner')

# Convert timestamp to datetime for proper x-axis handling
comparison_df["timestamp"] = pd.to_datetime(comparison_df["timestamp"])

# Optional: Downsample if plotting too slow
# comparison_df = comparison_df.iloc[::5, :]  # Uncomment to plot every 5th point

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(comparison_df["timestamp"], comparison_df["AUS_CZE"], label="Actual Capacity", color="blue")
ax.plot(comparison_df["timestamp"], comparison_df["AUS_CZE_pred"], label="Predicted Capacity",
        color="red", linestyle="dashed")

# Format x-axis dates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

# Labels and title
ax.set_xlabel("Timestamp")
ax.set_ylabel("Cross-Border Capacity")
ax.set_title("Predicted vs. Actual Cross-Border Transmission Capacity")
ax.legend()
ax.grid()

# Save figure
if save_plot:
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("Figure saved at:", fig_path)

plt.show()
