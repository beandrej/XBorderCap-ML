import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config
#from train_reg import TRAINING_SET, MODEL_NAME
import data_loader
from plotter_class import *
from train_reg import *

MODEL_NAME = 'BaseModel'
TRAINING_SET = 'BL_NTC_FULL'
BORDER_TYPE = 'NTC'
LOSS = 'MSELoss'


# Paths


MAXBEX_BORDERS = ["AUS_CZE", "CZE_AUS", "AUS_GER", "GER_AUS","BEL_FRA","FRA_BEL","BEL_GER","GER_BEL","BEL_NET","NET_BEL","CZE_GER",
                "CZE_POL","POL_CZE","GER_NET","NET_GER","GER_POL","POL_GER","GER_FRA","FRA_GER"]

NTC_BORDERS = [
        'AT_to_IT_NORD', 'IT_NORD_to_AT',
        'AT_to_CH', 'CH_to_AT', 'BE_to_GB', 'GB_to_BE', 'BE_to_NL', 'NL_to_BE',
        'SI_to_IT_NORD', 'IT_NORD_to_SI', 'DK_1_to_DE_LU', 'DE_LU_to_DK_1',
        'DK_1_to_NL', 'NL_to_DK_1', 'DK_2_to_DE_LU', 'DE_LU_to_DK_2',
        'ES_to_FR', 'FR_to_ES', 'ES_to_PT', 'PT_to_ES', 'FR_to_GB', 'GB_to_FR',
        'FR_to_IT_NORD', 'IT_NORD_to_FR', 'FR_to_CH', 'CH_to_FR',
        'GB_to_NL', 'NL_to_GB', 'DE_LU_to_NL', 'NL_to_DE_LU',
        'DE_LU_to_CH', 'CH_to_DE_LU', 'IT_NORD_to_CH', 'CH_to_IT_NORD',
        'NL_to_NO_2', 'NO_2_to_NL'
    ]

if BORDER_TYPE == 'NTC':
    split_col = NTC_BORDERS[0]
    BORDERS = NTC_BORDERS
elif BORDER_TYPE == 'MAXBEX':
    split_col = MAXBEX_BORDERS[0]
    BORDERS = MAXBEX_BORDERS
else: 
    raise ValueError("Wrong Border Type!")









PLOT_BORDER = True
PLOT_COMPARE_METRIC = False

if PLOT_BORDER:
    pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions_csv', f"pred_{MODEL_NAME}_{TRAINING_SET}_{LOSS}.csv")
    y_true_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{TRAINING_SET}.csv")
    plotObj = PredictionPlot(pred_path, y_true_path, split_col)

    for border in BORDERS:
        fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'results/plots/pred/{BORDER_TYPE}/{MODEL_NAME}', f"{border}_{MODEL_NAME}_{TRAINING_SET}_{LOSS}.png")
        plotObj.plot_border(border, fig_path, save_plot=True, show_plot=False)

"""
************************************************************************************
    COMPARISON PLOT
************************************************************************************
"""


if PLOT_COMPARE_METRIC:
    compareMetrix = CompareDFMetricsPlot(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_metrics/{MODEL_NAME}'), MODEL_NAME)
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/plots/model_metrics/{MODEL_NAME}')

    compareMetrix.compareValR2(os.path.join(base_path, f'comparison_ValR2_{MODEL_NAME}'), save_plot=True, show_plot=False)
    compareMetrix.compareTrainR2(os.path.join(base_path, f'comparison_TrainR2_{MODEL_NAME}'), save_plot=True, show_plot=False)

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
