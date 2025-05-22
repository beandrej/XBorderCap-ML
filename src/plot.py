import sys; sys.dont_write_bytecode = True
from utils.plot_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

models = ['BaseModel', 'Net', 'LSTM']
datasets = ["BL_FBMC_NORM", 'BL_NTC_NORM', 'FX_FBMC_NORM', 'FX_NTC_NORM']   
#ntcDatasets = ["FX_NTC_NORM", "BL_NTC_NORM"]
border_types = ['FBMC', 'NTC']
dataset_types = ['BL']

import pandas as pd
import glob
import matplotlib.pyplot as plt


# # Data
# labels = ['Classification', 'Regression']
# sizes = [69, 31]  # Proportional sizes
# display_values = [22, 10]  # Custom values to display inside slices
# colors = ['mediumseagreen', 'lightsalmon']

# # Plot
# fig, ax = plt.subplots()
# wedges, texts = ax.pie(
#     sizes,
#     labels=None,
#     colors=colors,
#     startangle=90
# )

# # Add custom values manually inside each wedge
# for i, wedge in enumerate(wedges):
#     angle = (wedge.theta2 + wedge.theta1) / 2
#     x = 0.7 * np.cos(np.radians(angle))
#     y = 0.7 * np.sin(np.radians(angle))
#     ax.text(x, y, str(display_values[i]), ha='center', va='center', fontsize=14)

# # Title
# plt.title('Border division for Hybrid model', fontweight='bold')

# # Legend
# ax.legend(
#     wedges,
#     labels,
#     loc="center right",
#     bbox_to_anchor=(1.15, 0.55)
# )

# # Make it a circle
# ax.axis('equal')
# plt.title('Border division in NTC domain', fontweight='bold', fontsize=16)
# plt.tight_layout()
# plt.show()
# for model in models:
#     #for dataset in datasets:
#     for border_type in border_types:
#         path = os.path.join(config.PROJECT_ROOT, f'src/results/model_metrics/{border_type}/{model}/metrics*.csv')
#         print(path)
#         # Step 1: Load all CSV files
#         csv_files = glob.glob(path)  # adjust path
#         if not csv_files:
#             #print(f"No CSV files found for model={model}, dataset={dataset}")
#             continue  # Skip this combination

#         # Step 2: Aggregate all dataframes
#         df_list = [pd.read_csv(f) for f in csv_files]

#         # Step 3: Stack them by epoch
#         all_df = pd.concat(df_list)

#         # Step 4: Group by epoch and compute the mean
#         mean_df = all_df.groupby('epoch')[['train_r2', 'val_r2', 'train_mae', 'val_mae']].mean().reset_index()

#         # Step 5: Plot using your function
#         #plotR2OverEpochs(mean_df, title=f"Average R² over Epochs Across Borders for {model} on {dataset}")
#         plotMaeOverEpochs(mean_df, title=f"Average MAE over Epochs Across Borders for {model} for {border_type}")
#         plt.show()

# border_types = ['FBMC', 'NTC']  # adjust as needed
# linestyles = {'FBMC': '-', 'NTC': '--'}
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = {
#     'train': default_colors[0],  # basic blue
#     'val': default_colors[1]     # basic orange
# }
# for model in models:
#     plt.figure(figsize=(10, 6))

#     for border_type in border_types:
#         path = os.path.join(config.PROJECT_ROOT, f'src/results/model_metrics/{border_type}/{model}', 'metrics*.csv')
#         csv_files = glob.glob(path)

#         if not csv_files:
#             print(f"No CSV files found for {model} in {border_type}")
#             continue

#         # Load and combine all files
#         df_list = [pd.read_csv(f) for f in csv_files]
#         all_df = pd.concat(df_list)

#         # Compute average metrics by epoch
#         mean_df = all_df.groupby('epoch')[['train_mae', 'val_mae']].mean().reset_index()
#         mean_df = mean_df[mean_df['epoch'] <= 40]

#         # Plot Train MAE
#         plt.plot(mean_df['epoch'], mean_df['train_mae'],
#                  linestyle=linestyles[border_type],
#                  color=colors['train'],
#                  label=f'{border_type} - Train MAE')

#         # Plot Val MAE
#         plt.plot(mean_df['epoch'], mean_df['val_mae'],
#                  linestyle=linestyles[border_type],
#                  color=colors['val'],
#                  label=f'{border_type} - Val MAE')

#     plt.xlabel('Epoch')
#     plt.ylabel('MAE')
#     plt.title(f"Train & Val MAE over Epochs for {model}")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
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

#for dataset in datasets:
    #plotHybridR2Bar(dataset)
    #plotHybridAccuracyBar(dataset) 
    #plotHybridMAEBar(dataset)
    
# for bordertype in border_types:
#     for model in models:
#         for datasettype in dataset_types:
#             plotMaxR2PerBorder(bordertype, model, datasettype)
#             plotMinMaePerBorder(bordertype, model, datasettype)


# for bordertype in border_types:
#     for model in models:
#         plotAvgTrainValAccOverTime(bordertype, model)

# df_full = pd.read_csv("C:/Users/andre/github/XBorderCap-ML/prep_data/old/BL_NTC_FULL.csv")
# df_norm = pd.read_csv("C:/Users/andre/github/XBorderCap-ML/prep_data/BL_FBMC_NORM.csv")

# #aus_ita_full = df_full["FRA_GBR"]
# aus_ita_norm = df_norm["GER_AUS"]

# num_points = len(aus_ita_norm)
# dates = pd.date_range(start="2022-06-09", end="2024-11-14", periods=num_points)

# # Split indices
# train_end = int(num_points * 0.76)
# val_end = int(num_points * 0.95)

# # Plot normalized data
# plt.figure(figsize=(16, 5))
# plt.plot(dates, aus_ita_norm, color='black', label="GER_AUS (Normalized)")
# plt.axvspan(dates[0], dates[train_end], color='green', alpha=0.1, label='Training')
# plt.axvspan(dates[train_end], dates[val_end], color='orange', alpha=0.1, label='Validation')
# plt.axvspan(dates[val_end], dates[-1], color='blue', alpha=0.1, label='Prediction/Future')
# plt.title("GER_AUS Time Series Split (Normalized)")

# plt.ylabel("Flow (Normalized)")
# plt.legend()
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# plt.tight_layout()
# plt.show()

# # Plot original (unnormalized) data
# plt.figure(figsize=(16, 5))
# plt.plot(dates, aus_ita_full, color='black', label="AUS_CZE")
# plt.axvspan(dates[0], dates[train_end], color='green', alpha=0.1, label='Training')
# plt.axvspan(dates[train_end], dates[val_end], color='orange', alpha=0.1, label='Validation')
# plt.axvspan(dates[val_end], dates[-1], color='blue', alpha=0.1, label='Prediction/Future')
# plt.title("AUS_CZE Time Series Split")

# plt.ylabel("Flow (MW)")
# plt.legend()
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# plt.tight_layout()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

fbmc = pd.read_csv("C:/Users/andre/github/XBorderCap-ML/prep_data/targets/FBMC.csv")
ntc = pd.read_csv("C:/Users/andre/github/XBorderCap-ML/prep_data/targets/NTC.csv")
# Load your datasets
fbmc = pd.read_csv("C:/Users/andre/github/XBorderCap-ML/prep_data/targets/FBMC.csv", index_col='timestamp', parse_dates=True)
ntc = pd.read_csv("C:/Users/andre/github/XBorderCap-ML/prep_data/targets/NTC.csv", index_col='timestamp', parse_dates=True)

# Merge on timestamp index (inner join)
merged = pd.merge(fbmc, ntc, left_index=True, right_index=True)
merged = pd.read_csv("C:/Users/andre/github/XBorderCap-ML/prep_data/BL_FBMC_NORM.csv", index_col='timestamp', parse_dates=True)

columns_of_interest = ['GER_FRA', 'GER_BEL', 'GER_NET', 'GER_generation_wind_onshore']
corr_matrix = merged[columns_of_interest].corr()
label_map = {
    'GER_FRA': 'GER_FRA',
    'GER_BEL': 'GER_BEL',
    'GER_NET': 'GER_NET',
    'GER_generation_wind_onshore': 'GER onshore wind'
}
corr_matrix.rename(index=label_map, columns=label_map, inplace=True)


# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Germany Export correlation with wind generation")
plt.tight_layout()
plt.show()
