import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config
import plotly

class PredictionPlot():
    def __init__(self, pred_path, true_path, target_col_start):
        full_df = pd.read_csv(true_path, index_col=0)
        target_idx = full_df.columns.get_loc(target_col_start)
        y_true = full_df.iloc[:, target_idx:].reset_index()
        y_pred = pd.read_csv(pred_path)
        
        self.comp_df = pd.merge(y_true, y_pred, on="timestamp", how='inner')
        self.comp_df["timestamp"] = pd.to_datetime(self.comp_df["timestamp"])

    def plot_border(self, border, save_path, save_plot=False, grid_on=True, show_plot=True):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.comp_df["timestamp"], self.comp_df[border],            label="Actual Capacity", color="blue")
        ax.plot(self.comp_df["timestamp"], self.comp_df[f"{border}_pred"],  label="Predicted Capacity", color="red", linestyle="dashed")
        
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Cross-Border Capacity [MW]")
        ax.set_title(f"{border} Predicted vs. Actual Capacity")
        ax.legend(loc=1)

        if grid_on:
            ax.grid()
        if save_plot:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Figure saved at:", save_path)
        if show_plot:
            plt.show()
    
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
        train_r2_scores = {}
        val_r2_scores = {}

        for dataset_name, filename in self.metrics_files.items():
            df = pd.read_csv(os.path.join(self.dir, filename))
            val_r2_scores[dataset_name] = df["val_r2"].tolist()
        
        plt.figure(figsize=(10, 6))

        for dataset, r2_list in val_r2_scores.items():
            plt.plot(r2_list, label=f"{dataset}")


        plt.title(f"Validation R² over Epochs ({self.model})")
        plt.xlabel("Epoch")
        plt.ylabel("R² Score")
        plt.legend(title="Dataset")
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
        val_r2_scores = {}

        for dataset_name, filename in self.metrics_files.items():
            df = pd.read_csv(os.path.join(self.dir, filename))
            train_r2_scores[dataset_name] = df["train_r2"].tolist()
        
        plt.figure(figsize=(10, 6))

        for dataset, r2_list in train_r2_scores.items():
            plt.plot(r2_list, label=f"{dataset}")


        plt.title(f"Training R² over Epochs ({self.model})")
        plt.xlabel("Epoch")
        plt.ylabel("R² Score")
        plt.legend(title="Dataset")
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
