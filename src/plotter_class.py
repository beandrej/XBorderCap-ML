import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config
import plotly

class PltCombinedBLDF():
    def __init__(self, pred_path, og_path, target_col):
        full_df = pd.read_csv(og_path, index_col=0)
        target_idx = full_df.columns.get_loc(target_col)
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
    
class PltModelMetric():
    def __init__(self, metrics_path, save_path):
        self.save_path = save_path
        self.df = pd.read_csv(metrics_path)
    
    def plotR2(self, save_plot=False, grid_on=True, show_plot=True):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df["epoch"], self.df["r2"], label="R2-Score", color="blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R2-Score")
        ax.set_title("R2-Score over epochs")
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

