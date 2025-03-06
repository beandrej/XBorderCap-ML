import os
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import config
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

# TODO rewrite read_target_variable() and read_feature_data() to iterate through a list of countries
# TODO write TargetData class to load set of target vars

class BaseData(Dataset):
    def __init__(self, df_name):
        self.name = df_name
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{self.name}.csv")

        if os.path.exists(self.path):
            self.loadCSV()
        else:
            raise ValueError("File not existing")

    def plotCorr(self):
        df = pd.DataFrame(self.data)
        plt.figure(figsize=(15, 14))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def printStats(self):
        print(self.data.info())
        print(self.data.head())

    def castType(self, datatype):
        self.data = self.data.astype(datatype)

    def loadCSV(self):
        print(f"\n Loading dataset: {self.name}")

        df = pd.read_csv(self.path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors='raise')
        self.data = df

        total_nans = df.isna().sum().sum()
        print(f"Total NaNs in dataset: {total_nans}")
        print(f"Data successfully loaded! Shape: {df.shape}")
        return df
    
    def saveCSV(self):
        self.data.index = pd.to_datetime(self.data.index)
        self.data.to_csv(self.path, index=True)
        print(f"Data saved as CSV to: {self.path}")
   
    def cutDataset(self, start_date="2019-01-01 00:00:00", end_date="2024-12-31 23:00:00"):
        print(f"\n Trimming dataset from {start_date} to {end_date}")

        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.loc[start_date:end_date]

        print(f"New range: {self.data.index.min()} to {self.data.index.max()}")

    def addTimeFeatures(self):
        self.data["hour"] = self.data.index.hour
        self.data["dayofweek"] = self.data.index.dayofweek  # Monday=0, Sunday=6
        self.data["month"] = self.data.index.month  # 1-12
        self.data["dayofyear"] = self.data.index.dayofyear  # 1-365
        self.data["weekofyear"] = self.data.index.isocalendar().week.astype(int)  # 1-52
        self.data["is_weekend"] = (self.data.index.dayofweek >= 5).astype(int) 
    
    def dropSparse(self, nan_threshold=0.5, zero_threshold=0.1):
        sparse_columns = self.data.columns[self.data.isna().mean() > nan_threshold]
        zero_columns = self.data.columns[(self.data == 0).mean() > zero_threshold]
        columns_to_drop = sparse_columns.union(zero_columns)
        self.data.drop(columns=columns_to_drop, inplace=True)

        if len(columns_to_drop) > 0:
            print(f"Dropping {len(sparse_columns)} sparse columns and {len(zero_columns)} zero-columns")
            print(f"Dropped Columns: {list(columns_to_drop)}")

    def dropEX(self):
        ex_columns = [col for col in self.data.columns if col.startswith("EX_")]
        if len(ex_columns) > 0:
            self.data.drop(columns=ex_columns, inplace=True)
            print(f"Dropped {len(ex_columns)} columns starting with 'EX_'")
            print(f"Dropped Columns: {ex_columns}")
        else:
            print("No 'EX_' columns found to drop.")

class CrossBorderData(BaseData):
    def __init__(self, c1, c2, domain, df_name, loadCSV=True):
        self.name = df_name
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{self.name}.csv")

        if loadCSV == True and os.path.exists(self.path):
            self.loadCSV()
        else:
            country1 = self.merge_country(c1)
            country2 = self.merge_country(c2)
            self.data = country1.merge(country2, how='outer', on='timestamp')
            self.drop_sparse()
            self.convert_to_float32()
            self.cut_dataset()

            target = self.merge_target_var(domain, c1, c2)
            self.data = target.merge(self.data, how='outer', on='timestamp')
            self.data = self.data.sort_values(by='timestamp')

            self.addTimeFeatures()

            if domain == 'max_bex':
                self.cut_dataset(start_date='2022-06-09 00:00:00', end_date='2024-12-31 23:00:00')
            else: 
                self.cut_dataset(start_date='2019-01-01 00:00:00', end_date='2024-11-30 23:00:00')

            print(self.data.info())

            self.data = self.data.ffill()
            self.data = self.data.bfill()
            
            self.saveCSV()

    def merge_country(self, country):    
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables')
        data = {}

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file == f'{country}.csv':
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, parse_dates=True, index_col=0, date_format='%Y-%m-%d %H:%M:%S')
                    feature_type = os.path.basename(root)
                    data[feature_type] = df

        combined_df = pd.concat(data.values(), axis=1)
        combined_df = combined_df.add_prefix(f"{country}_")

        return combined_df

    def merge_target_var(self, domain, c1, c2):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/02_target_variables', f"{domain}", f"{c1}_to_{c2}.csv")
        df = pd.read_csv(path, index_col=1)    
        df = df.drop(columns=[df.columns[0]]) 
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df.columns = [f"{c1}_{c2}_cross_border_capacity"]
        df = df.astype(np.float32)

        return df

    def add_installed_cap(self, hourly_df, country):
        
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables/entsoe/installed_capacity', f"{country}_installed_generation_capacity.csv")
        
        df = pd.read_csv(path, index_col=1)
        df = df.drop(columns=[df.columns[0]])
        df.index = pd.to_datetime(df.index) 
        df.index = df.index.tz_localize(None)
        df['year'] = df.index.year
        df = df.set_index('year')

        hourly_df['year'] = hourly_df.index.year
        merged_df = hourly_df.merge(df, on='year', how='left')
        merged_df = merged_df.set_index(hourly_df.index)
        merged_df = merged_df.drop(columns=['year'])

        return merged_df
    
class TypeData(BaseData):
    def __init__(self, source, type, df_name, loadCSV=True, saveCSV=False):
        self.name = df_name
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{self.name}.csv")
        self.source = source
        self.type = type

        if loadCSV and os.path.exists(self.path):
            self.loadCSV()
        else:
            self.data = self.merge_all()
            self.dropSparse()
            self.dropEX()
            self.cutDataset()

            if source.lower() == 'entsoe':
                self.remove_nordics()

            self.data = self.data.sort_values(by='timestamp')
            self.data = self.data.ffill()
            self.data = self.data.bfill()

            if saveCSV:
                self.saveCSV()

    def merge_all(self):

        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables', self.source, self.type)
        demand_data = []

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.csv'): 
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
                    country_code = file.split('.')[0]
                    if self.source.lower() == 'entsoe':
                        df = df.add_prefix(f"{country_code}_")
                    else:
                        df = df.add_suffix(f"_actual_load")
                    demand_data.append(df)

        combined_df = pd.concat(demand_data, axis=1)
        return combined_df
    
    def remove_nordics(self):
        nordics = ["DK1", "DK2", "EST", "FIN", "LAT", "LIT", "NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4"]
        nordic_data = [col for col in self.data.columns if col.startswith(tuple(nordics))]
        print(f"\nRemoving {len(nordic_data)} Nordic country columns from ENTSO-E dataset\n")
        self.data.drop(columns=nordic_data, inplace=True)

class TargetData(BaseData):
    def __init__(self, domain, df_name, loadCSV=True, saveCSV=False):
        self.name = df_name
        self.domain = domain
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{self.name}.csv")

        if loadCSV and os.path.exists(self.path):
            self.loadCSV()
        else:
            self.data = self.merge_all()
            self.dropSparse()
            self.cutDataset()
            #self.dropEX()
            self.data = self.data.sort_values(by='timestamp')
            self.data = self.data.ffill()
            self.data = self.data.bfill()

            if saveCSV:
                self.saveCSV()

    def merge_all(self):

        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/02_target_variables', self.domain)
        data = []

        if self.domain == 'max_bex':
            valid_countries = ["AUS", "BEL", "BKN", "CZE", "FRA", "GER", "NET", "POL"]
        elif self.domain == 'ntc':
            valid_countries = ["AUS", "BEL", "BKN", "DK1", "DK2", "ESP", "FRA", "GBR", "GER", "IT1", "NET", "NO2", "POR", "SWI"]
        
        country_pairs = list(itertools.permutations(valid_countries, 2))
        country_pairs.sort(key=lambda pair: (min(pair), max(pair)))
        filenames = [f"{c1}_to_{c2}.csv" for c1, c2 in country_pairs]
        valid_filenames = []

        for c1, c2 in country_pairs:
            filename = f"{c1}_to_{c2}.csv"
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                valid_filenames.append(filename)

        for file in valid_filenames:
            filepath = os.path.join(base_path, file)
            df = pd.read_csv(filepath, index_col=1, parse_dates=True)
            df.index = df.index.tz_localize(None)
            df.drop(columns=["Unnamed: 0"], inplace=True)
            data.append(df)

        merged_df = pd.concat(data, axis=1, join="outer")
        return merged_df

# class DataLoader(BaseData):
#     def __init__(self, df_name):
#         self.name = df_name
#         self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{self.name}.csv")

#         if os.path.exists(self.path):
#             self.loadCSV()
#         else:
#             raise ValueError("Dataset not existing")

        



