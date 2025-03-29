import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_class import *
import seaborn as sns

NTC_START = '2019-01-01 00:00:00'
NTC_END = '2024-12-10 23:00:00'

MAXBEX_START = '2022-06-09 00:00:00'
MAXBEX_END = '2024-12-31 23:00:00'

def main():
    """
    #*****************************************************************************************************************
    #   CREATOR
    #*****************************************************************************************************************
    """


    #created = TypeData('nordpool', 'generation_by_type', 'GEN_NP_AGG', loadCSV=False, saveCSV=True)
    # created.printStats()
    # created.cutDataset(start_date='2020-01-01 00:00:00', end_date=NTC_END)
    # created.printStats()

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

    #dem_tot = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "DEM_TOT.csv"), index_col=0)

    # network = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables/network', "combined_network.csv"), index_col=0)
    # dem_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "DEM_TOT.csv"), index_col=0)

    # network.index = pd.to_datetime(network.index, errors='coerce')
    # dem_df.index = pd.to_datetime(dem_df.index, errors='coerce')

    # network.index.name = 'timestamp'
    # dem_df.index.name = 'timestamp'

    # network = network.reset_index()
    # dem_df = dem_df.reset_index()

    # network["month_key"] = pd.to_datetime(network["timestamp"]).dt.to_period("M").dt.to_timestamp()
    # dem_df["month_key"] = pd.to_datetime(dem_df["timestamp"]).dt.to_period("M").dt.to_timestamp()

    # merged = pd.merge(dem_df, network, on="month_key", how="right")

    # # Drop month_key and reset timestamp as index
    # merged.drop(columns=["month_key"], inplace=True)
    # merged = merged.set_index("timestamp_x")
    # merged.index.name = "timestamp"
    # merged.index = pd.to_datetime(merged.index)
    # merged = merged.dropna(subset=["AUS_actual_load"])
    # merged.to_csv('aa.csv')

    # pop = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables/country', "population.csv"))
    # area = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables/country', "area.csv"))

    # merged = pd.merge(area, pop, left_on="zoneName", right_on="zone", how='inner')
    # merged = merged.drop(columns=["zone"])
    # merged.to_csv('sss.csv')


    # dem_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "DEM_TOT.csv"), index_col=0)
    # country_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables/country', "combined_country.csv"), index_col=0)
    # area_dict = dict(zip(country_df['zoneName'], country_df['area']))
    # pop_dict = dict(zip(country_df['zoneName'], country_df['population']))

    # for col in dem_df.columns:
    #     for country in area_dict:
    #         if col.startswith(country):
    #             dem_df[f'{country}_LOAD_PER_AREA'] = dem_df[col] / area_dict[country]
    #             dem_df[f'{country}_LOAD_PER_PP'] = dem_df[col] / pop_dict[country] * 1000
    
    # dem_df.to_csv('ayayay.csv')


    # new_df = df[['zoneName', 'area']]

    # print(new_df.info())

    # df1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "DEM_TOT_AREA_NTWRK.csv"))
    # df2 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "GENLOAD_AGG.csv"))

    # merged_df = pd.merge(df1, df2, how="inner", on="timestamp", suffixes=("", "_dup"))
    # merged_df = merged_df[[col for col in merged_df.columns if not col.endswith("_dup")]]
    # #merged_df = merged_df.drop(columns=["timestamp_y"])
    # merged_df = merged_df.set_index("timestamp")
    # merged_df.to_csv('aaa.csv')

    # new_df = new_df[~new_df["zoneName"].isin(["BKN", "RMB", "EX_SVK", "EX_HUN"])]

    # print(new_df.info())
    # new_df.to_csv('upadted.csv')



    """
    #*****************************************************************************************************************
    #   LOADER
    #*****************************************************************************************************************
    """


    # df = BaseData('GENLOAD_GNLD_AREA_TIME')
    # df.addTimeFeatures()
    # df.saveCSV()
    # df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "BASELINE_MAXBEX_GENLOAD_AGG.csv"), index_col=0)


    # selected_columns = ['GER_INFLEX', 'GER_IEX']
    # new_df = df[selected_columns]

    # # Optionally, save the new DataFrame to a new CSV file
    # new_df.to_csv("filtered_data.csv") 
    # #df.printStats()
    
    #.loadCSV()
    #df.info()
    ## # Find columns with NaNs
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

    # df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "MAX_BEX_old.csv"), index_col=0)

    # columns_to_keep = [
    #     "AUS_CZE", "CZE_AUS", "AUS_GER", "GER_AUS", "BEL_FRA", "FRA_BEL",
    #     "BEL_GER", "GER_BEL", "BEL_NET", "NET_BEL", "CZE_GER", "GER_CZE",
    #     "CZE_POL", "POL_CZE", "GER_NET", "NET_GER", "GER_POL", "POL_GER",
    #     "GER_FRA", "FRA_GER"
    # ]
    # new_df = df[columns_to_keep]
    # new_df.to_csv("filtered_dataframe.csv", index=True)

    #df = BaseData('GENLOAD_AGG_CTRY_RESLOAD')

    """
    #*****************************************************************************************************************
    #   MERGER
    #*****************************************************************************************************************
    """
    # df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "GENLOAD_GNLD.csv"), index_col=0)
    # df = df.drop(columns=[col for col in df.columns if col.endswith("_actual_load")])
    # df.to_csv('aaa.csv')



    # gen_ee = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "GENLOAD_GNLD_AREA_TIME.csv"), index_col=0)
    # gen_np = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "WEATHER.csv"), index_col=0)
    # dem_ee = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "NTC.csv"), index_col=0)
    # # # dem_np = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "NTC.csv"), index_col=0)

    # common_index = gen_ee.index.intersection(gen_np.index)
    # common_index = common_index.intersection(dem_ee.index)
    # # # common_index = common_index.intersection(dem_np.index)

    # gen_ee = gen_ee.loc[common_index]
    # gen_np = gen_np.loc[common_index]
    # dem_ee = dem_ee.loc[common_index]
    # # # dem_np = dem_np.loc[common_index]

    # merged_df = pd.concat([gen_ee, gen_np, dem_ee], axis=1)
    # output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "BL_NTC_FULL.csv")
    # merged_df.to_csv(output_path)
    

    """
    #***********************************************
    # ADD RES LOAD & IEX
    #***********************************************"
    """

    # df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "GENLOAD_INFLEX.csv"), index_col=0)

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


    #     # Calculate RES_LOAD
    #     if actual_load_col in df.columns and inflex_col in df.columns:
    #         res_load_col = f"{country}_RES_LOAD"
    #         df[res_load_col] = df[actual_load_col] - df[inflex_col]
    #         res_load_columns[country] = res_load_col

    #         # ✅ RES_LOAD_RATIO = RES_LOAD / actual_load
    #         if actual_load_col in df.columns:
    #             df[f"{country}_RES_LOAD_RATIO"] = df[res_load_col] / df[actual_load_col]

    #         # ✅ RES_LOAD_GRAD = difference (signed) between timesteps
    #         df[f"{country}_RES_LOAD_GRAD"] = df[res_load_col].diff().fillna(0)

    # # # Reorder columns
    # df.to_csv("updated_dataframe.csv")
    # print("Updated dataframe saved as 'updated_dataframe4.csv'")

    """
    #*****************************************************************************************************************
    #   PLOT
    #*****************************************************************************************************************
    """
    # country = "FRA"

    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "NTC.csv"), index_col=0)
    df.index = pd.to_datetime(df.index)
    df_resampled = df.resample('h').mean()

    dfData = BaseData('BASELINE_MAXBEX_WITH_NTC_GENLOAD_AGG')

    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "BASELINE_MAXBEX_WITH_NTC_GENLOAD_AGG.csv"), index_col=0)
    #selected_columns = ['FRA_IEX', 'GER_IEX', 'BEL_IEX', 'GER_FRA', 'GER_BEL', 'BEL_FRA', 'FR_to_ES', 'FR_to_CH', 'DE_LU_to_CH']
    selected_columns = ['GER_RES_LOAD_GRAD', 'GER_FRA', 'GER_BEL', 'DE_LU_to_CH']

    sns.heatmap(df[selected_columns].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_resampled['timestamp'], )
    ax.set_ylabel("Average MAXBEX Cross Border Capacities [MWh]")
    ax.set_title("Average MAXBEX Cross-Border Capacities \nWeekday vs Weekend")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    #plot_cols(df, selected_columns)


    enable_plot = False

    if enable_plot:
        plot_cols(dfData, columns=selected_columns, start_date="2015-01-01", end_date="2026-01-01")


def plot_cols(dataset, columns=None, column_indices=None, start_date=None, end_date=None):

    df = dataset.data

    if column_indices is not None:
        index_based_columns = [df.columns[i] for i in column_indices if i < len(df.columns)]
    else:
        index_based_columns = []


    selected_columns = set(columns or []) | set(index_based_columns)
    available_cols = [col for col in selected_columns if col in df.columns]

    complete_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h') 
    plot_data = df[available_cols].reindex(complete_index)  

    # If specific start and end dates are provided, filter the data
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
