import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config
#from train_reg import TRAINING_SET, MODEL_NAME
import data_loader
from plotter_class import *
import train_FBMC

MODEL_NAME = 'LSTM'
TRAINING_SET = 'BL_FBMC_FULL'
BORDER_TYPE = 'MAXBEX'
LOSS = 'SmoothL1Loss'

PCA_COMP = train_FBMC.PCA_COMP
SEQ_LEN = train_FBMC.SEQ_LEN


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









PLOT_BORDER = False
PLOT_COMPARE_METRIC = False

if PLOT_BORDER:
    if MODEL_NAME == 'LSTM':
        pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions_csv', f"pred_{MODEL_NAME}_{TRAINING_SET}_{LOSS}_{SEQ_LEN}.csv")
    else:
        pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions_csv', f"pred_{MODEL_NAME}_{TRAINING_SET}_{LOSS}.csv")

    y_true_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{TRAINING_SET}.csv")
    plotObj = PredictionPlot(pred_path, y_true_path, split_col)

    for border in BORDERS:
        if MODEL_NAME == 'LSTM':
            fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/plots/pred/{BORDER_TYPE}/{MODEL_NAME}', f"{border}_{MODEL_NAME}_{TRAINING_SET}_{LOSS}_{SEQ_LEN}.png")
        else:
            fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/plots/pred/{BORDER_TYPE}/{MODEL_NAME}', f"{border}_{MODEL_NAME}_{TRAINING_SET}_{LOSS}.png")
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


# Load your DataFrames
df1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_metrics/MAXBEX/LSTM/SEQ_LEN=24/metrics_LSTM_AGG_FBMC_SmoothL1Loss_24_PCA128.csv'))
df3 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_metrics/MAXBEX/LSTM/SEQ_LEN=24/metrics_LSTM_AGG_FBMC_SmoothL1Loss_24_NOPCA.csv'))
df2 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_metrics/MAXBEX/LSTM/SEQ_LEN=24/metrics_LSTM_BL_FBMC_FULL_SmoothL1Loss_24_PCA128.csv'))
df4 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_metrics/MAXBEX/LSTM/SEQ_LEN=24/metrics_LSTM_BL_FBMC_FULL_SmoothL1Loss_24_NOPCA.csv'))

plt.figure(figsize=(12, 6))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:grey']

# df1
plt.plot(df1['train_r2'], label='Train R2 | PCA=128 | AGG_FBMC', color=colors[0], linestyle='-')
plt.plot(df1['val_r2'], label='Val R2   | PCA=128 | AGG_FBMC', color=colors[0], linestyle='--')

# df2
plt.plot(df2['train_r2'], label='Train R2 | PCA=128 | BL_FBMC_FULL', color=colors[1], linestyle='-')
plt.plot(df2['val_r2'], label='Val R2    | PCA=128 | BL_FBMC_FULL', color=colors[1], linestyle='--')

# df3
plt.plot(df3['train_r2'], label='Train R2 | PCA=None | AGG_FBMC', color=colors[2], linestyle='-')
plt.plot(df3['val_r2'], label='Val R2   | PCA=None | AGG_FBMC', color=colors[2], linestyle='--')

# df4
plt.plot(df4['train_r2'], label='Train R2 | PCA=None | BL_FBMC_FULL', color=colors[3], linestyle='-')
plt.plot(df4['val_r2'], label='Val R2   | PCA=None | BL_FBMC_FULL', color=colors[3], linestyle='--')


# Style
plt.title("Train vs Val R² — Across Models & PCA Settings")
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


