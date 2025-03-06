import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_class import *
import seaborn as sns

# TODO JOIN TARGET DF WITH FEATURE DF TO PLOT CORR

DATASET_NAME = 'DEM_EE'

# X-BorderSet
COUNTRY1 = 'AUS'
COUNTRY2 = 'SWI'
DOMAIN = 'ntc'

# TypeSet
SOURCE = 'entsoe'
DATATYPE = 'demand'


enable_plot = False

def main():


    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "MAX_BEX_WITH_FEATURES.csv"), index_col=0)
    print(df.nunique())
    print(df.nunique()[df.nunique() < 5])
    print("TOTAL NaN")
    print(df.isna().sum().sum())

    plt.figure(figsize=(15, 14))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.show()


    if enable_plot:
        plot(df, column_indices=[0, 1, 3], start_date="2015-01-01", end_date="2026-01-01")

def plot(dataset, columns=None, column_indices=None, start_date=None, end_date=None):

    df = dataset.data  # **Directly use the DataFrame from dataset**

    # **Convert column indices to column names**
    if column_indices is not None:
        index_based_columns = [df.columns[i] for i in column_indices if i < len(df.columns)]
    else:
        index_based_columns = []

    # **Combine manually selected columns with indexed columns**
    selected_columns = set(columns or []) | set(index_based_columns)
    available_cols = [col for col in selected_columns if col in df.columns]

    if not available_cols:
        print("❌ No valid columns selected for plotting.")
        return

    print(f"📊 Plotting columns: {available_cols}")

    # **Filter data by date range**
    plot_data = df[available_cols].copy()
    if start_date and end_date:
        plot_data = plot_data.loc[start_date:end_date]

    # **Plot selected columns**
    plt.figure(figsize=(12, 6))
    for col in available_cols:
        plt.plot(plot_data.index, plot_data[col], label=col)

    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.title("Comparison of Selected Features")
    plt.legend()
    plt.grid(True)
    plt.show()

def combine(df_list, out_name):

    data = []
    for single_df in df_list:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{single_df}.csv")
        df = pd.read_csv(path, index_col=0)
        data.append(df)

    df = pd.concat(data, axis=1)

    df = df.ffill()
    df = df.bfill()
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "WEATHER.csv"))
    print(df.info())

if __name__ == "__main__":
    main()
