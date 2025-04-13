import os
import sys; sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.plot_utils import *

MODEL_NAME = 'Net'
TRAINING_SET = 'FX_FBMC_NORM'
BORDER_TYPE = TRAINING_SET.split('_')[1]
NORMALIZING = TRAINING_SET.split('_')[2] == 'NORM'


MAXBEX_BORDERS = ["AUS_CZE", "CZE_AUS", "AUS_GER", "GER_AUS","BEL_FRA","FRA_BEL","BEL_GER","GER_BEL","BEL_NET","NET_BEL","CZE_GER",
                "CZE_POL","POL_CZE","GER_NET","NET_GER","GER_POL","POL_GER","GER_FRA","FRA_GER"]

NTC_BORDERS = [
"AUS_ITA","ITA_AUS","AUS_SWI","SWI_AUS","BEL_GBR","GBR_BEL","SVN_ITA","ITA_SVN",
"DK1_GER","GER_DK1","DK1_NET","NET_DK1","DK2_GER","GER_DK2","ESP_FRA","FRA_ESP",
"ESP_POR","POR_ESP","FRA_GBR","GBR_FRA","FRA_ITA","ITA_FRA","FRA_SWI","SWI_FRA",
"GBR_NET","NET_GBR","GER_SWI","SWI_GER","ITA_SWI","SWI_ITA","NET_NO2","NO2_NET"
]

if BORDER_TYPE == 'NTC':
    split_col = NTC_BORDERS[0]
    BORDERS = NTC_BORDERS
elif BORDER_TYPE == 'FBMC':
    split_col = MAXBEX_BORDERS[0]
    BORDERS = MAXBEX_BORDERS
else: 
    raise ValueError("Wrong Border Type!")


PLOT_BORDER = True
PLOT_COMPARE_METRIC = False

if PLOT_BORDER:
    if MODEL_NAME == 'LSTM':
        pred_path = os.path.join(config.PROJECT_ROOT, f'src/results/predictions_csv/{MODEL_NAME}', f"pred_{MODEL_NAME}_{TRAINING_SET}_{config.SEQ_LEN}.csv")
    else:
        pred_path = os.path.join(config.PROJECT_ROOT, f'src/results/predictions_csv/{MODEL_NAME}', f"pred_{MODEL_NAME}_{TRAINING_SET}.csv")

    y_true_path = os.path.join(config.PROJECT_ROOT, 'prep_data', f"{TRAINING_SET}.csv")
    plotObj = PredictionPlot(pred_path, y_true_path, split_col)

    for border in BORDERS:
        if MODEL_NAME == 'LSTM':
            fig_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/pred/{BORDER_TYPE}/{MODEL_NAME}', f"{border}_{MODEL_NAME}_{TRAINING_SET}_{config.SEQ_LEN}.html")
        else:
            fig_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/pred/{BORDER_TYPE}/{MODEL_NAME}', f"{border}_{MODEL_NAME}_{TRAINING_SET}.html")

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


