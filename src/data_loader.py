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

    #*****************************************************************************************************************
    #   CREATOR
    #*****************************************************************************************************************

    # created = TypeData('nordpool', 'generation_by_type', 'GROUPED_GEN_NP', loadCSV=False, saveCSV=True)
    # created.printStats()

    # combine(df_list=[loaded1, loaded2], out_name='TOT_GEN_AGG')
    # print(df.nunique())
    # print(df.nunique()[df.nunique() < 5])
    # print("TOTAL NaN")
    # print(df.isna().sum().sum())


    # created = BaseData('TOT_AGG_FEATURES')
    # created.printStats()
    # created.addTimeFeatures()
    # created.printStats()
    # created.saveCSV()

    #*****************************************************************************************************************
    #   LOADER
    #*****************************************************************************************************************

    df = BaseData('cleaned_dataframe')
    # # Find columns with NaNs
    # nan_columns = df.isna().sum()

    # # Filter only columns where NaNs exist
    # nan_columns = nan_columns[nan_columns > 0]

    # # Print columns with their NaN counts
    # print("Columns with NaN values and their counts:")
    # print(nan_columns)

    #loaded = BaseData('NTC')

    # df_cleaned = df.dropna()
    # # Print the new shape after removing NaNs
    # print(f"New shape after dropping NaNs: {df_cleaned.shape}")

    # # Save the cleaned DataFrame to a CSV file
    # df_cleaned.to_csv("cleaned_dataframe.csv", index=False)

    # print("Cleaned dataframe saved as 'cleaned_dataframe.csv'")

    #*****************************************************************************************************************
    #   MERGER
    #*****************************************************************************************************************

    # df1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "TOT_AGG_FEATURES.csv"), index_col=0)
    # df2 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "NTC.csv"), index_col=0)
    # merged_load = pd.merge(df1, df2, left_index=True, right_index=True, how="outer")
    # merged_load.info()
    # merged_load.to_csv("updated_dataframe444.csv")




    # countries = set(col.split('_')[0] for col in df.columns)

    # # Dictionary to store new columns before inserting them in order
    # iex_columns = {}
    # res_load_columns = {}

    # for country in countries:
    #     tot_gen_col = f"{country}_TOT_GEN"
    #     actual_load_col = f"{country}_actual_load"
    #     inflex_col = f"{country}_INFLEX"

    #     # Calculate IEX if the necessary columns exist
    #     if tot_gen_col in df.columns and actual_load_col in df.columns:
    #         df[f"{country}_IEX"] = df[tot_gen_col] - df[actual_load_col]
    #         iex_columns[country] = f"{country}_IEX"

    #     # Calculate RES_LOAD if the necessary columns exist
    #     if tot_gen_col in df.columns and inflex_col in df.columns:
    #         df[f"{country}_RES_LOAD"] = df[tot_gen_col] - df[inflex_col]
    #         res_load_columns[country] = f"{country}_RES_LOAD"

    # # Reorder columns
    # df.to_csv("updated_dataframe.csv")
    # print("Updated dataframe saved as 'updated_dataframe4.csv'")
    # gen1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "GROUPED_GEN_EE.csv"), index_col=0)
    # gen2 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "GROUPED_GEN_NP.csv"), index_col=0)



    # plt.figure(figsize=(15, 14))
    # sns.heatmap(df.corr(), annot=False, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    # plt.title("Feature Correlation Heatmap")
    # plt.show()


    if enable_plot:
        plot_cols(loaded, column_indices=[0, 1, 3], start_date="2015-01-01", end_date="2026-01-01")


def plot_cols(dataset, columns=None, column_indices=None, start_date=None, end_date=None):

    df = dataset.data

    if column_indices is not None:
        index_based_columns = [df.columns[i] for i in column_indices if i < len(df.columns)]
    else:
        index_based_columns = []

    selected_columns = set(columns or []) | set(index_based_columns)
    available_cols = [col for col in selected_columns if col in df.columns]

    plot_data = df[available_cols].copy()
    if start_date and end_date:
        plot_data = plot_data.loc[start_date:end_date]


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
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{out_name}.csv"))
    print(df.info())

if __name__ == "__main__":
    main()
