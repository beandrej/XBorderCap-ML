import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class PredictionPlot():
    def __init__(self, pred_path, true_path, target_col_start):
        full_df = pd.read_csv(true_path, index_col=0)
        target_idx = full_df.columns.get_loc(target_col_start)
        y_true = full_df.iloc[:, target_idx:].reset_index()
        y_pred = pd.read_csv(pred_path)
        
        self.comp_df = pd.merge(y_true, y_pred, on="timestamp", how='inner')
        self.comp_df["timestamp"] = pd.to_datetime(self.comp_df["timestamp"])

    def plot_border(self, border, save_path=None, save_plot=False, show_plot=True):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.comp_df["timestamp"],
            y=self.comp_df[border],
            mode='lines',
            name='Actual Capacity',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=self.comp_df["timestamp"],
            y=self.comp_df[f"{border}_pred"],
            mode='lines',
            name='Predicted Capacity',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f"{border} Predicted vs. Actual Capacity",
            xaxis_title="Timestamp",
            yaxis_title="Cross-Border Capacity [MW]",
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
            template='plotly_white'
        )

        if save_plot and save_path:
            fig.write_html(save_path.replace(".png", ".html"))
            print("Interactive plot saved at:", save_path.replace(".png", ".html"))

        if show_plot:
            fig.show()

class SingleMetricPlot():
    def __init__(self, model, dataset, metrics_path, save_path):
        self.model = model
        self.dataset = dataset
        self.save_path = save_path
        self.df = pd.read_csv(metrics_path)
    
    def plotR2(self, save_plot=False, grid_on=True, show_plot=True):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df["epoch"], self.df["train_r2"], label="Training R2-Score", color="blue")
        ax.plot(self.df["epoch"], self.df["val_r2"], label="Validation R2-Score", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R2-Score")
        ax.set_title(f"Training and Validation R2-Score of {self.model} on {self.dataset}")
        ax.legend()
        if grid_on:
            ax.grid()
        if save_plot:
            plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
            print("Figure saved at:", self.save_path)
        if show_plot:
            plt.show()
    
    def plotTrainValLoss(self, save_plot=False, grid_on=True, show_plot=True):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df["epoch"], self.df["train_loss"], label="Training Loss", color="blue")
        ax.plot(self.df["epoch"], self.df["val_loss"], label="Validation Loss", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("MSE of Training vs Validation loss")
        ax.legend()
        if grid_on:
            ax.grid()
        if save_plot:
            plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
            print("Figure saved at:", self.save_path)
        if show_plot:
            plt.show()

class CompareDFMetricsPlot():
    def __init__(self, dir, model):
        self.dir = dir
        self.model = model
        self.metrics_files = {}
        for f in os.listdir(dir):
            if f.startswith(f"metrics_{model}"):
                dataset_name = f.replace(f"metrics_{model}_", "").replace(".csv", "")
                self.metrics_files[dataset_name] = f

    def compareValR2(self, save_path=None, save_plot=False, grid_on=True, show_plot=True):
        val_r2_scores = {}

        for dataset_name, filename in self.metrics_files.items():
            df = pd.read_csv(os.path.join(self.dir, filename))
            val_r2_scores[dataset_name] = df["val_r2"].tolist()

        # Extract unique dataset bases and assign colors
        dataset_base_names = sorted(set(name.rsplit('_', 1)[0] for name in val_r2_scores))
        loss_types = sorted(set(name.rsplit('_', 1)[-1] for name in val_r2_scores))

        color_map = {name: color for name, color in zip(dataset_base_names, plt.cm.tab10.colors)}
        line_styles = ['-', '--', '-.', ':']  # Up to 4 line styles
        line_style_map = {loss: style for loss, style in zip(loss_types, line_styles)}

        plt.figure(figsize=(12, 7))

        for full_name, r2_list in val_r2_scores.items():
            dataset_base, loss_type = full_name.rsplit('_', 1)
            color = color_map.get(dataset_base, 'gray')
            linestyle = line_style_map.get(loss_type, '-')
            plt.plot(r2_list, label=full_name, color=color, linestyle=linestyle)

        plt.title(f"Validation R² over Epochs ({self.model})")
        plt.xlabel("Epoch")
        plt.ylabel("R² Score")
        plt.legend(title="Dataset + Loss", bbox_to_anchor=(1.05, 1), loc='upper left')
        if grid_on:
            plt.grid(True)
        plt.tight_layout()
        if save_plot and save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()

    def compareTrainR2(self, save_path=None, save_plot=False, grid_on=True, show_plot=True):
        train_r2_scores = {}

        for dataset_name, filename in self.metrics_files.items():
            df = pd.read_csv(os.path.join(self.dir, filename))
            train_r2_scores[dataset_name] = df["train_r2"].tolist()

        # Extract base dataset names and loss types
        dataset_base_names = sorted(set(name.rsplit('_', 1)[0] for name in train_r2_scores))
        loss_types = sorted(set(name.rsplit('_', 1)[-1] for name in train_r2_scores))

        # Assign consistent colors and line styles
        color_map = {name: color for name, color in zip(dataset_base_names, plt.cm.tab10.colors)}
        line_styles = ['-', '--', '-.', ':']
        line_style_map = {loss: style for loss, style in zip(loss_types, line_styles)}

        plt.figure(figsize=(12, 7))

        for full_name, r2_list in train_r2_scores.items():
            dataset_base, loss_type = full_name.rsplit('_', 1)
            color = color_map.get(dataset_base, 'gray')
            linestyle = line_style_map.get(loss_type, '-')
            plt.plot(r2_list, label=full_name, color=color, linestyle=linestyle)

        plt.title(f"Training R² over Epochs ({self.model})")
        plt.xlabel("Epoch")
        plt.ylabel("R² Score")
        plt.legend(title="Dataset + Loss", bbox_to_anchor=(1.05, 1), loc='upper left')
        if grid_on:
            plt.grid(True)
        plt.tight_layout()
        if save_plot and save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()

def plotActualVsPredicted(df, timestamp_col, actual_col, pred_col, title=None, figsize=(12, 5)):
    """
    Plots actual vs predicted values over time.
    df: DataFrame with predictions and ground truth
    timestamp_col: column name for the x-axis (usually a datetime index or column)
    actual_col: name of the actual values column
    pred_col: name of the predicted values column
    """
    plt.figure(figsize=figsize)
    plt.plot(df[timestamp_col], df[actual_col], label='Actual', linewidth=2)
    plt.plot(df[timestamp_col], df[pred_col], label='Predicted', linestyle='--', alpha=0.8)
    plt.title(title or f"Actual vs Predicted - {actual_col}")
    plt.xlabel("Timestamp")
    plt.ylabel("Capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotR2MaeOverEpochs(df, title=None, figsize=(12, 5)):
    """
    Plots R² and MAE (train & validation) across epochs.
    df: DataFrame with columns: 'epoch', 'train_r2', 'val_r2', 'train_global_mae', 'val_global_mae'
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].plot(df['epoch'], df['train_r2'], label='Train R²')
    axs[0].plot(df['epoch'], df['val_r2'], label='Val R²')
    axs[0].set_title('R² Over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('R² Score')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df['epoch'], df['train_global_mae'], label='Train MAE')
    axs[1].plot(df['epoch'], df['val_global_mae'], label='Val MAE')
    axs[1].set_title('MAE Over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAE')
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle(title or "Training Progress")
    plt.tight_layout()
    plt.show()

def compareModelsR2MaeSummary(metrics_dict, title_prefix="Model Comparison"):
    """
    Plots min MAE and max R² for multiple models.
    metrics_dict: dict with model name -> DataFrame with train/val R² and MAE per epoch
    """
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
    """
    Heatmap of classification accuracy or regression R² across borders and models (for Hybrid).
    df: DataFrame with one row per (border, model) combination
    """
    heatmap_data = df.pivot(index=index_col, columns=columns_col, values=value_col)
    plt.figure(figsize=(10, len(heatmap_data)*0.5))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(title or f"{value_col} by Border and Model")
    plt.tight_layout()
    plt.show()

def compareDatasetsBoxplot(df, metric_col='val_r2', group_col='dataset_type', hue_col='model', title=None):
    """
    Boxplot comparison for BL vs FX or other dataset categories.
    df: DataFrame with 'dataset_type', 'model', and metric columns
    """
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
  