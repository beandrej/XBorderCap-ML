import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config
from train_reg import TRAINING_SET, MODEL_NAME
import data_loader
from plotter_class import *


# Paths
pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results/predictions_csv', f"pred_{MODEL_NAME}_{TRAINING_SET}.csv")
fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results/plots/pred/baseModel', f"{MODEL_NAME}_{TRAINING_SET}.png")
y_true_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '../prep_data', f"{TRAINING_SET}.csv")

borders = ["AUS_CZE", "CZE_AUS", "AUS_GER", "GER_AUS","BEL_FRA","FRA_BEL","BEL_GER","GER_BEL","BEL_NET","NET_BEL","CZE_GER",
          "CZE_POL","POL_CZE","GER_NET","NET_GER","GER_POL","POL_GER","GER_FRA","FRA_GER"]

plotObj = PltCombinedBLDF(pred_path, y_true_path, "AUS_CZE")

for border in borders:

    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results/plots/pred/baseModel', f"{border}_{MODEL_NAME}_{TRAINING_SET}.png")
    plotObj.plot_border(border, fig_path, save_plot=True, show_plot=False)

bl_borders = PltCombinedBLDF(pred_path, y_true_path, "AUS_CZE", fig_path)
bl_borders.plot_border("AUS_CZE", save_plot=True)

# nn_metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                         'results/model_metrics', f"metrics_{MODEL_NAME}_{TRAINING_SET}.csv")
# metrics_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                        'results/plots/model_metrics', f"{MODEL_NAME}_{TRAINING_SET}.png")

# nn_metrics = PltModelMetric(nn_metrics_path, metrics_save_path)
# nn_metrics.plotR2(save_plot=True)
# nn_metrics.plotTrainValLoss(save_plot=True)

# # Load actual Y data
# full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                    '../prep_data', "BASELINE_MAXBEX.csv"), index_col=0)
# first_target_idx = full_df.columns.get_loc("AUS_CZE")
# Y = full_df.iloc[:, first_target_idx:].reset_index()

# # Load predictions
# predictions_df = pd.read_csv(pred_path)

# # Merge on timestamp
# comparison_df = pd.merge(Y, predictions_df, on="timestamp", how='inner')

# # Convert timestamp to datetime for proper x-axis handling
# comparison_df["timestamp"] = pd.to_datetime(comparison_df["timestamp"])

# # Optional: Downsample if plotting too slow
# # comparison_df = comparison_df.iloc[::5, :]  # Uncomment to plot every 5th point

# # Plotting
# fig, ax = plt.subplots(figsize=(14, 6))
# ax.plot(comparison_df["timestamp"], comparison_df["AUS_CZE"], label="Actual Capacity", color="blue")
# ax.plot(comparison_df["timestamp"], comparison_df["AUS_CZE_pred"], label="Predicted Capacity",
#         color="red", linestyle="dashed")

# # Format x-axis dates
# ax.xaxis.set_major_locator(mdates.AutoDateLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# fig.autofmt_xdate()

# # Labels and title
# ax.set_xlabel("Timestamp")
# ax.set_ylabel("Cross-Border Capacity")
# ax.set_title("Predicted vs. Actual Cross-Border Transmission Capacity")
# ax.legend()
# ax.grid()

# # Save figure
# if save_plot:
#     os.makedirs(os.path.dirname(fig_path), exist_ok=True)
#     plt.savefig(fig_path, dpi=300, bbox_inches='tight')
#     print("Figure saved at:", fig_path)

# plt.show()
