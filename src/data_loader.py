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

NTC_START = '2019-01-01 00:00:00'
NTC_END = '2024-12-10 23:00:00'

MAXBEX_START = '2022-06-09 00:00:00'
MAXBEX_END = '2024-12-31 23:00:00'

def main():

    #*****************************************************************************************************************
    #   CREATOR
    #*****************************************************************************************************************



    #created = TypeData('nordpool', 'generation_by_type', 'GEN_NP', loadCSV=False, saveCSV=True)
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




    #*****************************************************************************************************************
    #   LOADER
    #*****************************************************************************************************************



    #df = BaseData('BASELINE_NTC')
    #df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "BASELINE_MAXBEX_NTC.csv"), index_col=0)


    # selected_columns = ['GER_generation_wind_onshore', 'GER_generation_wind_offshore', 'DE_LU_to_CH']
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



    #*****************************************************************************************************************
    #   MERGER
    #*****************************************************************************************************************



    # df1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "plots_df.csv"), index_col=0)
    # df2 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "NTC.csv"), index_col=0)
    # merged = pd.merge(df1, df2, left_index=True, right_index=True, how="outer")
    # merged.info()
    # merged.to_csv("updated_dataframe444.csv")

    # gen_ee = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "GENLOAD.csv"), index_col=0)
    # gen_np = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "WEATHER.csv"), index_col=0)
    # dem_ee = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "MAXBEX.csv"), index_col=0)
    # dem_np = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "NTC.csv"), index_col=0)

    # common_index = gen_ee.index.intersection(gen_np.index)
    # common_index = common_index.intersection(dem_ee.index)
    # common_index = common_index.intersection(dem_np.index)

    # gen_ee = gen_ee.loc[common_index]
    # gen_np = gen_np.loc[common_index]
    # dem_ee = dem_ee.loc[common_index]
    # dem_np = dem_np.loc[common_index]

    # merged_df = pd.concat([gen_ee, gen_np, dem_ee, dem_np], axis=1)
    # output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "BASELINE_MAXBEX_NTC.csv")
    # merged_df.to_csv(output_path)
    
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



    #*****************************************************************************************************************
    #   PLOT
    #*****************************************************************************************************************

    country = "FRA"

    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "BASELINE_MAXBEX_NTC.csv"), index_col=0)
    df.index = pd.to_datetime(df.index)
    df_resampled = df.resample('h').mean()


    df_resampled['is_weekend'] = (df_resampled.index.weekday >= 5).astype(int)
    columns_of_interest = ["BEL_FRA", "GER_POL", "GER_FRA"]
    df_resampled['sum_selected'] = df_resampled[columns_of_interest].sum(axis=1)

    sum_weekday_avg = df_resampled[df_resampled['is_weekend'] == 0]['sum_selected'].mean()
    sum_weekend_avg = df_resampled[df_resampled['is_weekend'] == 1]['sum_selected'].mean()

    print(f"Weekday Sum Avg: {sum_weekday_avg:.2f}")
    print(f"Weekend Sum Avg: {sum_weekend_avg:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Weekday', 'Weekend'], [sum_weekday_avg, sum_weekend_avg], color=['skyblue', 'orange'])
    ax.set_ylabel("Average MAXBEX Cross Border Capacities [MWh]")
    ax.set_title("Average MAXBEX Cross-Border Capacities \nWeekday vs Weekend")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


    if enable_plot:
        plot_cols(df, columns=[], start_date="2015-01-01", end_date="2026-01-01")


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
