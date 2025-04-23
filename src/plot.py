import os
import sys; sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.plot_utils import *

models = ['Net', 'LSTM', 'BaseModel', 'Hybrid']
datasets = ["FX_NTC_NORM", "BL_NTC_NORM", "BL_FBMC_NORM", "FX_FBMC_NORM"]   
ntcDatasets = ["FX_NTC_NORM", "BL_NTC_NORM"]
border_types = ['FBMC', 'NTC']
dataset_types = ['BL', 'FX']

# for dataset in datasets:
#     if dataset.split('_')[1] == 'FBMC':
#         for border in config.FBMC_BORDERS:
#             predictionCompareModel(dataset, border)
#     else:
#         for border in config.NTC_BORDERS:
#             predictionCompareModel(dataset, border)

# for dataset in datasets:
#     barMAETestMetrics(dataset)
#     barR2TestMetrics(dataset)
#     barAvgR2(dataset)
#     barAvgMAE(dataset)

# for dataset in datasets:
#     plotHybridR2Bar(dataset)
    
# for bordertype in border_types:
#     for model in models:
#         for datasettype in dataset_types:
#             plotMaxR2PerBorder(bordertype, model, datasettype)
#             plotMinMaePerBorder(bordertype, model, datasettype)


for bordertype in border_types:
    for model in models:
        plotAvgTrainValAccOverTime(bordertype, model)