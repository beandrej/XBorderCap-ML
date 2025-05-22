import os
import sys; sys.dont_write_bytecode = True
import numpy as np
import seaborn as sns
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

model_colors = {
    'Net': '#4C72B0',        # Deep Blue
    'BaseModel': '#DD8452',  # Soft Orange
    'LSTM': '#55A868',       # Teal Green
    'Hybrid': "#C44E52FF"      # Rich Red
}

def mergeModelPred(dataset):
    border_type = dataset.split('_')[1]

    true_path = os.path.join(config.PROJECT_ROOT, 'prep_data/targets', f"{border_type}_normed.csv")
    net_pred_path = os.path.join(config.PROJECT_ROOT, f'src/results/predictions_csv/Net', f'pred_Net_{dataset}.csv')
    bl_pred_path = os.path.join(config.PROJECT_ROOT, f'src/results/predictions_csv/BaseModel', f'pred_BaseModel_{dataset}.csv')
    lstm_pred_path = os.path.join(config.PROJECT_ROOT, f'src/results/predictions_csv/LSTM', f'pred_LSTM_{dataset}.csv')
    hybrid_pred_path = os.path.join(config.PROJECT_ROOT, f'src/results/predictions_csv/Hybrid', f'pred_Hybrid_{dataset}.csv')

    true_df = pd.read_csv(true_path, index_col=0, parse_dates=True)
    net_pred_df = pd.read_csv(net_pred_path, index_col=0, parse_dates=True)
    net_pred_df = net_pred_df.add_suffix('_Net')
    bl_pred_df = pd.read_csv(bl_pred_path, index_col=0, parse_dates=True)
    bl_pred_df = bl_pred_df.add_suffix('_BaseModel')
    lstm_pred_df = pd.read_csv(lstm_pred_path, index_col=0, parse_dates=True)
    lstm_pred_df = lstm_pred_df.add_suffix('_LSTM')
    hybrid_pred_df = pd.read_csv(hybrid_pred_path, index_col=0, parse_dates=True)
    hybrid_pred_df = hybrid_pred_df.add_suffix('_Hybrid')

    merged_df = pd.concat([true_df, net_pred_df, bl_pred_df, lstm_pred_df, hybrid_pred_df], axis=1, join='inner')
    return merged_df

def predictionCompareModel(dataset, border, figsize=(12, 5)):

    model_colors = {
        'Net': '#69b3a2',
        'BaseModel': '#e79675',
        'LSTM': '#8da0cb',
        'Hybrid': '#e78ac3',
        'TCN': '#a6d854'
    }

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/pred/model_comparison/{dataset}')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'pred_modelComp_{dataset}_{border}.png')

    df = mergeModelPred(dataset)

    plt.figure(figsize=figsize)
    plt.plot(df.index, df[f'{border}'], label='Actual', linewidth=2, color='black', alpha=0.6)
    plt.plot(df.index, df[f'{border}_pred_Net'], label='Net', linestyle='-.', color=model_colors['Net'], alpha=1)
    plt.plot(df.index, df[f'{border}_pred_BaseModel'], label='BaseModel', linestyle='-.', color=model_colors['BaseModel'], alpha=1)
    plt.plot(df.index, df[f'{border}_pred_LSTM'], label='LSTM', linestyle='-.', color=model_colors['LSTM'], alpha=1)
    plt.plot(df.index, df[f'{border}_pred_Hybrid'], label='Hybrid', linestyle='-.', color=model_colors['Hybrid'], alpha=1)
    plt.plot(df.index, df[f'{border}_pred_TCNHybrid'], label='TCN', linestyle='-.', color=model_colors['TCN'], alpha=1)
    plt.title(f"Actual vs Predicted Capacity {dataset} - {border}")
    plt.xlabel("Timestamp")
    plt.ylabel("Normalized Capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def mergeFXBL(border_type, model):
    true_path = os.path.join(config.PROJECT_ROOT, 'prep_data/targets', f"{border_type}_normed.csv")
    fx_pred_path = os.path.join(config.PROJECT_ROOT, f'src/results/predictions_csv/{model}', f'pred_{model}_FX_{border_type}_NORM.csv')
    bl_pred_path = os.path.join(config.PROJECT_ROOT, f'src/results/predictions_csv/{model}', f'pred_{model}_BL_{border_type}_NORM.csv')

    true_df = pd.read_csv(true_path, index_col=0, parse_dates=True)
    fx_pred_df = pd.read_csv(fx_pred_path, index_col=0, parse_dates=True)
    fx_pred_df = fx_pred_df.add_suffix('_FX')
    bl_pred_df = pd.read_csv(bl_pred_path, index_col=0, parse_dates=True)
    bl_pred_df = bl_pred_df.add_suffix('_BL')

    merged_df = pd.concat([true_df, fx_pred_df, bl_pred_df], axis=1, join='inner')
    return merged_df

def predictionCompareFXandBL(border_type, model, border, figsize=(12, 5)):

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/pred/FX_BL/{border_type}/{model}')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'pred_FXBL_{model}_{border_type}_{border}.png')


    df = mergeFXBL(border_type, model)
    plt.figure(figsize=figsize)
    plt.plot(df.index, df[f'{border}'], label='Actual', linewidth=2, color='black', alpha=0.6)
    plt.plot(df.index, df[f'{border}_pred_BL'], label='BL Dataset', linestyle='--', color='red', alpha=1)
    plt.plot(df.index, df[f'{border}_pred_FX'], label='FX Dataset', linestyle='--', color='blue', alpha=1)
    plt.title(f"Actual vs Predicted Capacity {model} - {border}")
    plt.xlabel("Timestamp")
    plt.ylabel("Normalized Capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()





def plotR2OverEpochs(df, title=None, figsize=(12, 5)):
    plt.figure(figsize=figsize)
    plt.plot(df['epoch'], df['train_r2'], label='Train R²')
    plt.plot(df['epoch'], df['val_r2'], label='Val R²')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title(title if title else 'R² Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotMaeOverEpochs(df, title=None, figsize=(12, 6)):
    df = df[df['epoch'] <= 40]  # filter up to epoch 40
    plt.figure(figsize=figsize)
    plt.plot(df['epoch'], df['train_mae'], label='Train MAE')
    plt.plot(df['epoch'], df['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title(title if title else 'MAE Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compareModelsR2MaeSummary(metrics_dict, title_prefix="Model Comparison"):
    summary = []
    for model_name, df in metrics_dict.items():
        summary.append({
            'model': model_name,
            'max_train_r2': df['train_r2'].max(),
            'max_val_r2': df['val_r2'].max(),
            'min_train_mae': df['train_global_mae'].min(),
            'min_val_mae': df['val_global_mae'].min()
        })

    df_summary = pd.DataFrame(summary)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=df_summary, x='model', y='max_val_r2', ax=axs[0])
    axs[0].set_title(f'{title_prefix} - Max Val R²')
    axs[0].set_ylabel('R²')
    axs[0].set_xlabel('Model')

    sns.barplot(data=df_summary, x='model', y='min_val_mae', ax=axs[1])
    axs[1].set_title(f'{title_prefix} - Min Val MAE')
    axs[1].set_ylabel('MAE')
    axs[1].set_xlabel('Model')

    plt.tight_layout()
    plt.show()

def plotHybridClassificationHeatmap(df, value_col='val_cls_acc', index_col='border', columns_col='model', title=None):
    heatmap_data = df.pivot(index=index_col, columns=columns_col, values=value_col)
    plt.figure(figsize=(10, len(heatmap_data)*0.5))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(title or f"{value_col} by Border and Model")
    plt.tight_layout()
    plt.show()

def compareDatasetsBoxplot(df, metric_col='val_r2', group_col='dataset_type', hue_col='model', title=None):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=group_col, y=metric_col, hue=hue_col)
    plt.title(title or f"{metric_col} by Dataset Type")
    plt.xlabel(group_col)
    plt.ylabel(metric_col)
    plt.legend(title=hue_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotLagCorr(df, target_col, max_lag=48, feature_subset=None):

    features = feature_subset or [col for col in df.columns if col != target_col]
    correlations = {}

    for lag in range(1, max_lag + 1):
        lagged_df = df[features + [target_col]].copy()
        lagged_df['lagged_target'] = lagged_df[target_col].shift(-lag)

        # Drop NaNs and constant columns
        lagged_df = lagged_df.dropna()
        valid_features = [col for col in features if lagged_df[col].std() > 1e-6]

        # Only use features with variance
        if len(valid_features) == 0:
            continue

        corr_series = lagged_df[valid_features].corrwith(lagged_df['lagged_target'])
        correlations[lag] = corr_series

    corr_df = pd.DataFrame(correlations).T  # [lag, feature]

    # Fill missing values (from dropped features at certain lags)
    corr_df = corr_df.fillna(0)

    # --- Plot ---
    plt.figure(figsize=(14, 6))
    im = plt.imshow(corr_df.T, aspect='auto', cmap='coolwarm', interpolation='none')
    plt.colorbar(im, label='Correlation')
    plt.title(f"Correlation of Features with Future '{target_col}'")
    plt.xlabel("Lag (time steps into the future)")
    plt.ylabel("Feature index")
    plt.xticks(ticks=np.arange(0, max_lag, step=4), labels=np.arange(1, max_lag + 1, step=4))
    plt.tight_layout()
    plt.show()

    return corr_df

def plotTrainValLoss(metrics_df, loss, dataset, model, saveFig=True, showPlot=False):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Training Loss", color="blue")
    ax.plot(metrics_df["epoch"], metrics_df["val_loss"], label="Validation Loss", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss {loss}")
    ax.set_title(f"{loss} of Training vs Validation loss of {model} on {dataset}")
    ax.legend()
    ax.grid()
    if saveFig: plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../{model}/results/plots/training/losses', f"{model}_{dataset}_{loss}_TrainValLoss.png"), dpi=300, bbox_inches='tight') 
    if showPlot: plt.show()
    else: plt.close()

def plotR2(metrics_df, dataset, model, loss, saveFig=True, showFig=False):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(metrics_df["epoch"], metrics_df["train_r2"], label="Training R2-Score", color="blue")
    ax.plot(metrics_df["epoch"], metrics_df["val_r2"], label="Validation R2-Score", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2-Score")
    ax.set_title(f"Training and Validation R2-Score of {model} on {dataset}")
    ax.legend()
    ax.grid()
    if saveFig: plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../{model}/results/plots/training/r2', f"{model}_{dataset}_{loss}_R2Score.png"), dpi=300, bbox_inches='tight')
    if showFig: plt.show()
    else: plt.close()     

def plotTrainValSplit(Y, val_start_idx, val_end_idx, test_start_idx, saveFig=True, showFig=False, border_name=None):
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../results/plots/training/train_val_split',
        f"{border_name}_NORM_train_val_split.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(14, 5))

    if isinstance(Y, pd.DataFrame) and Y.shape[1] > 1:
        Y_plot = Y.iloc[:, 0]
    else:
        Y_plot = Y.squeeze()

    plt.plot(Y_plot.values, label=f"{border_name}", color="black", linewidth=1)

    # Highlight segments
    plt.axvspan(0, val_start_idx, color='green', alpha=0.1, label='Train (Part 1)')
    plt.axvspan(val_start_idx, val_end_idx, color='orange', alpha=0.15, label='Validation')
    plt.axvspan(val_end_idx, test_start_idx, color='green', alpha=0.1, label='Train (Part 2)')
    plt.axvspan(test_start_idx, len(Y_plot), color='blue', alpha=0.08, label='Prediction/Future')

    plt.xlabel("Timestep")
    plt.ylabel("Flow (MW)")
    plt.title(f"{border_name or 'Border'} Time Series Split")
    plt.legend()
    plt.tight_layout()
    if saveFig: plt.savefig(save_path, dpi=300)
    if showFig: plt.show()
    else: plt.close()





def loadAndRename(path, suffix):
    df = pd.read_csv(path)
    df.columns = ['border'] + [col + suffix for col in df.columns[1:]]
    return df

def mergeTestMetricsCompareModels(dataset):
    net_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/Net', f'test_metrics_Net_{dataset}.csv')
    bl_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/BaseModel', f'test_metrics_BaseModel_{dataset}.csv')
    lstm_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/LSTM', f'test_metrics_LSTM_{dataset}.csv')
    hybrid_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/Hybrid', f'test_metrics_Hybrid_{dataset}.csv')
    tcn_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/TCNHybrid', f'test_metrics_TCNHybrid_{dataset}.csv')

    net_df = loadAndRename(net_path, '_Net')
    bl_df = loadAndRename(bl_path, '_BaseModel')
    lstm_df = loadAndRename(lstm_path, '_LSTM')
    hybrid_df = loadAndRename(hybrid_path, '_Hybrid')
    tcn_df = loadAndRename(hybrid_path, '_TCNHybrid')

    merged_df = net_df.merge(bl_df, on='border') \
                  .merge(lstm_df, on='border') \
                  .merge(hybrid_df, on='border')
    return merged_df

def barR2TestMetrics(dataset, clip_min=-1, clip_max=1, figsize=(24, 16)):
    df = mergeTestMetricsCompareModels(dataset)

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/testing/R2')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{dataset}_r2_comparison.png')

    r2_cols = [col for col in df.columns if col.lower().startswith('r2_')]
    df_r2 = df[['border'] + r2_cols].copy()

    for col in r2_cols:
        df_r2[col] = pd.to_numeric(df_r2[col], errors='coerce')

    # Calculate mean R² per border for sorting (ascending for worst to best)
    df_r2['avg_r2'] = df_r2[r2_cols].mean(axis=1)
    df_r2 = df_r2.sort_values(by='avg_r2', ascending=True)

    # Melt for plotting
    df_r2_melted = df_r2[['border'] + r2_cols].melt(id_vars='border', var_name='Model', value_name='R2')
    df_r2_melted['Model'] = df_r2_melted['Model'].apply(lambda x: x.split('_')[-1])

    # Use sorted borders as categorical order
    df_r2_melted['border'] = pd.Categorical(df_r2_melted['border'], categories=df_r2['border'], ordered=True)

    # Save original R² values before clipping for annotation
    df_r2_melted['R2_raw'] = df_r2_melted['R2']
    df_r2_melted['R2'] = df_r2_melted['R2'].clip(lower=clip_min, upper=clip_max)

    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df_r2_melted, x='R2', y='border', hue='Model', palette=model_colors)

    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title(f'R² Scores by Border and Model – {dataset}', fontsize=24, weight='bold')
    plt.xlabel('R² Score', fontsize=14)
    plt.ylabel('')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', fontsize=16)

    # X-axis limits
    min_val = df_r2_melted['R2'].min()
    max_val = df_r2_melted['R2'].max()
    plt.xlim(min_val - 0.01, max_val + 0.01)

    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def barMAETestMetrics(dataset, clip_min=0, clip_max=1, figsize=(24, 16)):
    df = mergeTestMetricsCompareModels(dataset)

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/testing/MAE')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{dataset}_mae_comparison.png')

    mae_cols = [col for col in df.columns if col.lower().startswith('mae_')]
    df_mae = df[['border'] + mae_cols].copy()

    for col in mae_cols:
        df_mae[col] = pd.to_numeric(df_mae[col], errors='coerce')

    # Calculate mean MAE per border for ordering
    df_mae['avg_mae'] = df_mae[mae_cols].mean(axis=1)
    df_mae = df_mae.sort_values(by='avg_mae', ascending=False)

    # Prepare for melting
    df_mae_melted = df_mae[['border'] + mae_cols].melt(id_vars='border', var_name='Model', value_name='MAE')
    df_mae_melted['Model'] = df_mae_melted['Model'].apply(lambda x: x.split('_')[-1])

    # Set categorical order by sorted border list
    df_mae_melted['border'] = pd.Categorical(df_mae_melted['border'], categories=df_mae['border'], ordered=True)

    # Clip for visibility
    df_mae_melted['MAE_raw'] = df_mae_melted['MAE']
    df_mae_melted['MAE'] = df_mae_melted['MAE'].clip(lower=clip_min, upper=clip_max)

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df_mae_melted, x='MAE', y='border', hue='Model', palette=model_colors)

    plt.title(f'MAE by Border and Model – {dataset}', fontsize=24, weight='bold')
    plt.xlabel('MAE', fontsize=14)
    plt.ylabel('')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)

    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', fontsize=16)

    # Adjust x-axis
    min_val = df_mae_melted['MAE'].min()
    max_val = df_mae_melted['MAE'].max()
    plt.xlim(min_val - 0.05, max_val + 0.05)

    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def barAvgMAE(dataset, figsize=(12,8)):
    df = mergeTestMetricsCompareModels(dataset)

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/testing/MAE')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'AVG_{dataset}_mae_comp.png')

    mae_cols = [col for col in df.columns if col.lower().startswith('mae_')]

    avg_mae = df[mae_cols].mean().rename(lambda x: x.split('_')[-1])
    avg_mae = avg_mae.reset_index().rename(columns={'index': 'Model', 0: 'MAE'})

    plt.figure(figsize=figsize)
    sns.barplot(data=avg_mae, x='Model', y='MAE', palette=model_colors)

    plt.title(f'Average MAE per Model – {dataset}', fontsize=16, weight='bold')
    plt.xlabel('')
    plt.ylabel('Average MAE', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Optional: add value labels
    for index, row in avg_mae.iterrows():
        plt.text(index, row['MAE'] + 0.005, f"{row['MAE']:.3f}", ha='center', fontsize=11)

    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def barAvgR2(dataset, clip_min=-1, clip_max=1, figsize=(12,8)):
    df = mergeTestMetricsCompareModels(dataset)

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/testing/R2')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'AVG_{dataset}_r2_comp.png')

    r2_cols = [col for col in df.columns if col.lower().startswith('r2_')]

    avg_r2 = df[r2_cols].mean().rename(lambda x: x.split('_')[-1])
    avg_r2 = avg_r2.reset_index().rename(columns={'index': 'Model', 0: 'R²'})

    # Save original for annotation, then clip
    avg_r2['R²_raw'] = avg_r2['R²']
    avg_r2['R²'] = avg_r2['R²'].clip(lower=clip_min, upper=clip_max)

    plt.figure(figsize=figsize)
    ax = sns.barplot(data=avg_r2, x='Model', y='R²', palette=model_colors)

    plt.title(f'Average R² per Model – {dataset}', fontsize=16, weight='bold')
    plt.xlabel('')
    plt.ylabel('Average R²', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Show real value if it was clipped
    for index, row in avg_r2.iterrows():
        display_val = row['R²_raw']
        y_pos = row['R²']
        offset = 0.02 if y_pos >= 0 else -0.04
        text = f"{display_val:.2f}" + ("*" if y_pos != display_val else "")
        plt.text(index, y_pos + offset, text, ha='center', fontsize=11)

    # Tight y-limits
    plt.ylim(clip_min - 0.05, clip_max + 0.05)

    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def plotHybridMAEBar(dataset):
    model = 'Hybrid'
    hybrid_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/BACKTEST/{model}', f'test_metrics_{model}_{dataset}.csv')   
    df = pd.read_csv(hybrid_path)
    df_sorted = df.sort_values(by='test_mae', ascending=True)

    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=df_sorted, x='test_mae', y='border', palette='viridis')

    plt.title(f"{model} Model MAE per Border - {dataset}", fontsize=16, weight='bold')
    plt.xlabel('MAE', fontsize=14)
    plt.ylabel('Border', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    for i, (acc, border) in enumerate(zip(df_sorted['test_mae'], df_sorted['border'])):
        plt.text(acc + 0.001, i, f"{acc:.2f}", va='center', fontsize=10)

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/{model}/MAE')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'BACKTEST_{model}_{dataset}_MAE.png')

    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def plotHybridAccuracyBar(dataset):
    model = 'Hybrid'
    hybrid_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/{model}', f'test_metrics_{model}_{dataset}.csv')   
    df = pd.read_csv(hybrid_path)
    # Filter to only classification rows (r2 == 0.0)
    df = df[df['test_r2'] == 0.0]
    df_sorted = df.sort_values(by='test_acc', ascending=True)

    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=df_sorted, x='test_acc', y='border', palette='viridis')

    plt.title(f"{model} Model Accuracy per Border - {dataset}", fontsize=16, weight='bold')
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Border', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    # Optionally annotate bars
    for i, (acc, border) in enumerate(zip(df_sorted['test_acc'], df_sorted['border'])):
        plt.text(acc + 0.01, i, f"{acc:.2f}", va='center', fontsize=10)

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/{model}/clsAcc')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{model}_{dataset}_clsAcc.png')

    plt.tight_layout()
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def plotHybridR2Bar(dataset):
    model = 'Hybrid'
    hybrid_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/BACKTEST/{model}', f'test_metrics_{model}_{dataset}.csv')   
    df = pd.read_csv(hybrid_path)

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/{model}/R2')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'BACKTEST_{model}_{dataset}_R2.png')

    df = df[df['test_r2'] != 0.0]
    df_sorted = df.sort_values(by='test_r2', ascending=True)

    plt.figure(figsize=(10, max(6, len(df_sorted) * 0.3)))
    ax = sns.barplot(data=df_sorted, x='test_r2', y='border', palette='viridis')

    plt.title(f'{model} Model R² per Border - {dataset}', fontsize=16, weight='bold')
    plt.xlabel('R² Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    
    
    for i, (r2, border) in enumerate(zip(df_sorted['test_r2'], df_sorted['border'])):
        offset = -0.05 if r2 >= 0 else 0.05  # opposite side of bar
        ha = 'right' if r2 >= 0 else 'left'
        plt.text(offset, i, f"{r2:.2f}", va='center', ha=ha, fontsize=11)

    plt.xlim(-1, 1)
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def pieHybridClsAcc(dataset):
    hybrid_path = os.path.join(config.PROJECT_ROOT, f'src/results/test_metrics/Hybrid', f'test_metrics_Hybrid_{dataset}.csv')   
    df = pd.read_csv(hybrid_path)

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/Hybrid/Acc')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{dataset}_ACC.png')

    cls_count = df[df['r2'] == 0.0].shape[0]
    reg_count = df[df['r2'] != 0.0].shape[0]

    labels = ['Classification', 'Regression']
    sizes = [cls_count, reg_count]
    colors = ['#66c2a5', '#fc8d62']


    def absolute_value(val):
        total = sum(sizes)
        count = int(round(val * total / 100))
        return f'{count}'

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct=absolute_value, startangle=90,
        textprops={'fontsize': 14})
    plt.axis('equal')
    plt.title('Border division for Hybrid model', fontsize=16, weight='bold')
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def plotAvgTrainValR2OverTime(bordertype, model, metric='r2'):

    metric_dir = os.path.join(config.PROJECT_ROOT, f'src/results/model_metrics/{bordertype}/{model}')
    files = glob.glob(os.path.join(metric_dir, '*.csv'))

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/training/overEpochs')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{model}_{bordertype}_epochsR2.png')

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    df_concat = pd.concat(dfs)
    avg_df = df_concat.groupby('epoch').agg({
        f'train_{metric}': 'mean',
        f'val_{metric}': 'mean'
    }).reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(avg_df['epoch'], avg_df[f'train_{metric}'], label='Train', linewidth=2)
    plt.plot(avg_df['epoch'], avg_df[f'val_{metric}'], label='Validation', linewidth=2)
    plt.title(f"Average {metric.upper()} Over Time - {model} on {bordertype}", fontsize=14, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def plotAvgTrainValAccOverTime(bordertype, model, metric='acc'):
    if model != 'Hybrid':
        return
    metric_dir = os.path.join(config.PROJECT_ROOT, f'src/results/model_metrics/{bordertype}/{model}')
    files = glob.glob(os.path.join(metric_dir, '*.csv'))

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/training/overEpochs')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{model}_{bordertype}_epochsAcc.png')

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    df_concat = pd.concat(dfs)

    avg_df = df_concat.groupby('epoch').agg({
        f'train_cls_{metric}': 'mean',
        f'val_cls_{metric}': 'mean'
    }).reset_index()


    plt.figure(figsize=(10, 5))

    plt.plot(avg_df['epoch'], avg_df[f'train_cls_{metric}'], label='Train', linewidth=2)
    plt.plot(avg_df['epoch'], avg_df[f'val_cls_{metric}'], label='Validation', linewidth=2)

    plt.title(f"Average {metric.upper()} Over Time - {model} on {bordertype}", fontsize=14, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def plotAvgTrainValMAEOverTime(bordertype, model, metric='mae'):

    metric_dir = os.path.join(config.PROJECT_ROOT, f'src/results/model_metrics/{bordertype}/{model}')
    files = glob.glob(os.path.join(metric_dir, '*.csv'))

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/training/overEpochs')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{model}_{bordertype}_epochsMAE.png')

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    df_concat = pd.concat(dfs)
    if model == 'Hybrid':
        avg_df = df_concat.groupby('epoch').agg({
            f'train_global_{metric}': 'mean',
            f'val_global_{metric}': 'mean'
        }).reset_index()
    else:
        avg_df = df_concat.groupby('epoch').agg({
            f'train_{metric}': 'mean',
            f'val_{metric}': 'mean'
        }).reset_index()

    plt.figure(figsize=(10, 5))
    if model == 'Hybrid':
        plt.plot(avg_df['epoch'], avg_df[f'train_global_{metric}'], label='Train', linewidth=2)
        plt.plot(avg_df['epoch'], avg_df[f'val_global_{metric}'], label='Validation', linewidth=2)
    else:
        plt.plot(avg_df['epoch'], avg_df[f'train_{metric}'], label='Train', linewidth=2)
        plt.plot(avg_df['epoch'], avg_df[f'val_{metric}'], label='Validation', linewidth=2)
    plt.title(f"Average {metric.upper()} Over Time - {model} on {bordertype}", fontsize=14, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()


def plotMaxR2PerBorder(bordertype, model, dataset_type):
    metric_dir = os.path.join(config.PROJECT_ROOT, f'src/results/model_metrics/{bordertype}/{model}')
    files = glob.glob(os.path.join(metric_dir, '*.csv'))
    data = []

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/training/R2')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{model}_{bordertype}_trainValMAXR2.png')

    data = []
    for file in files:
        base = os.path.basename(file)
        if model == 'Hybrid':
            border = '_'.join(base.split('_')[-3:-1]).replace('.csv', '')
        else:
            border = '_'.join(base.split('_')[-2:]).replace('.csv', '')
        df = pd.read_csv(file)
        data.append({
            'dataset': 'FX' if '_FX_' in file else 'BL',
            'border': border,
            'max_train_r2': df['train_r2'].max(),
            'max_val_r2': df['val_r2'].max()
        })
    df_plot = pd.DataFrame(data)
    df_plot = df_plot.sort_values(by='max_val_r2')

    plt.figure(figsize=(8,12))
    for dataset, marker in zip(['BL', 'FX'], ['o', '^']):
        subset = df_plot[df_plot['dataset'] == dataset]
        plt.scatter(subset['max_train_r2'], subset['border'], label=f'Train R² ({dataset})', color='#4C72B0', marker=marker)
        plt.scatter(subset['max_val_r2'], subset['border'], label=f'Val R² ({dataset})', color='#C44E52', marker=marker)
    plt.title(f'Max Train-R² vs Val-R² - {model} on {bordertype}', fontsize=14, weight='bold')
    plt.xlabel('R²', fontsize=16)
    plt.ylabel('Border', fontsize=16)
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.xlim(-1, 1)
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS: 
        plt.show()
    else:
        plt.close()

def plotMinMaePerBorder(bordertype, model, dataset_type):
    metric_dir = os.path.join(config.PROJECT_ROOT, f'src/results/model_metrics/{bordertype}/{model}')
    files = glob.glob(os.path.join(metric_dir, '*.csv'))
    data = []

    fig_base_path = os.path.join(config.PROJECT_ROOT, f'src/results/plots/training/MAE')
    os.makedirs(fig_base_path, exist_ok=True)
    fig_path = os.path.join(fig_base_path, f'{model}_{bordertype}_trainValMINMAE.png')
    
    for file in files:
        base = os.path.basename(file)
        if model == 'Hybrid':
            border = '_'.join(base.split('_')[-3:-1]).replace('.csv', '')
        else:
            border = '_'.join(base.split('_')[-2:]).replace('.csv', '')
        df = pd.read_csv(file)
        if model == 'Hybrid':
            data.append({
                'dataset': 'FX' if '_FX_' in file else 'BL',
                'border': border,
                'min_train_mae': df['train_global_mae'].min(),
                'min_val_mae': df['val_global_mae'].min()
            })    
        else:
            data.append({
                'dataset': 'FX' if '_FX_' in file else 'BL',
                'border': border,
                'min_train_mae': df['train_mae'].min(),
                'min_val_mae': df['val_mae'].min()
            })
    df_plot = pd.DataFrame(data)
    df_plot = df_plot.sort_values(by='min_val_mae')

    plt.figure(figsize=(8, 12))
    for dataset, marker in zip(['BL', 'FX'], ['o', '^']):
        subset = df_plot[df_plot['dataset'] == dataset]
        plt.scatter(subset['min_train_mae'], subset['border'], label=f'Train MAE ({dataset})', color='#4C72B0', marker=marker)
        plt.scatter(subset['min_val_mae'], subset['border'], label=f'Val MAE ({dataset})', color='#C44E52', marker=marker)

    plt.title(f'Min Train-MAE vs Val-MAE - {model} on {bordertype}', fontsize=14, weight='bold')
    plt.xlabel('MAE', fontsize=16)
    plt.ylabel('Border', fontsize=16)
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.xlim(0, 1)
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(fig_path)
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

