import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import config
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

# TODO write cutting function determining start/stop -> start stop of target variable too naive
# TODO rewrite read_target_variable() and read_feature_data() to iterate through a list of countries
# TODO fix the data XD
# TODO just start from scratch bro..

class CrossBorderData(Dataset):

    def __init__(self, c1, c2, domain, df_name, load_from_file=True, convert_to_tensor=True):
        
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{df_name}.pt")

        if load_from_file == True and os.path.exists(self.data_path):
            self.load_data()
        else:
            country1 = self.merge_country(c1)
            country2 = self.merge_country(c2)
            self.data = country1.merge(country2, how='outer', on='timestamp')
            self.drop_sparse()
            self.convert_to_float32()
            self.cut_dataset()
            self.imputer()

            target = self.merge_target_var(domain, c1, c2)
            self.data = target.merge(self.data, how='outer', on='timestamp')
            self.data = self.data.sort_values(by='timestamp')

            self.data["hour"] = self.data.index.hour
            self.data["dayofweek"] = self.data.index.dayofweek  # Monday=0, Sunday=6
            self.data["month"] = self.data.index.month  # 1-12
            self.data["dayofyear"] = self.data.index.dayofyear  # 1-365
            self.data["weekofyear"] = self.data.index.isocalendar().week.astype(int)  # 1-52
            self.data["is_weekend"] = (self.data.index.dayofweek >= 5).astype(int) 

            if domain == 'max_bex':
                self.cut_dataset(start_date='2022-06-09 00:00:00', end_date='2024-12-31 23:00:00')
            else: 
                self.cut_dataset(start_date='2019-01-01 00:00:00', end_date='2024-12-31 23:00:00')

            self.timestamp = self.data.index.astype(str).tolist()
            self.xborder_cap = self.data[f"{c1}_{c2}_cross_border_capacity"]

            print(self.data.info())

            if convert_to_tensor:
                self.to_tensor(c1, c2)
                self.save_data(c1, c2)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.timestamp[idx] 
    
    def to_tensor(self, c1, c2):
        self.X = torch.tensor(self.data.drop(columns=[f"{c1}_{c2}_cross_border_capacity"]).values, dtype=torch.float32)
        self.y = torch.tensor(self.data[f"{c1}_{c2}_cross_border_capacity"].values, dtype=torch.float32).view(-1, 1)

    def save_data(self, c1, c2):

        feature_columns = list(self.data.drop(columns=[f"{c1}_{c2}_cross_border_capacity"]).columns)
        save_dict = {
            "X": self.X,
            "y": self.y,
            "timestamp": self.timestamp,
            "feature_columns": feature_columns
        }
        torch.save(save_dict, self.data_path)
        print(f"Data saved to {self.data_path}")

    def load_data(self):

        print(f"\n Loading data from {self.data_path}...")

        data = torch.load(self.data_path, weights_only=True)

        self.X = data["X"]
        self.y = data["y"]
        self.timestamp = data["timestamp"]
        self.feature_columns = data.get("feature_columns", [f"feature_{i}" for i in range(self.X.shape[1])])

        print("\n Checking after loading data...")
        print("NaNs in X tensor:", torch.isnan(self.X).sum().item())
        print("NaNs in y tensor:", torch.isnan(self.y).sum().item())


    def convert_to_float32(self):
        for col in self.data.columns:
            if self.data[col].dtype in [np.float64, np.int64]:
                self.data[col] = self.data[col].astype(np.float32)

    def cut_dataset(self, start_date="2018-01-01 00:00:00", end_date="2024-12-31 23:00:00"):

        print(f"\n Trimming dataset from {start_date} to {end_date}")

        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.loc[start_date:end_date]

        print(f"New range: {self.data.index.min()} to {self.data.index.max()}")

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
      
    def drop_sparse(self, nan_threshold=0.5, zero_threshold=0.1):
        total_rows = len(self.data)

        sparse_columns = self.data.columns[self.data.isna().sum() > total_rows * nan_threshold]
        zero_columns = [col for col in self.data.columns if (self.data[col] == 0).all()]
        low_variance_columns = [col for col in self.data.columns if (self.data[col] != 0).sum() < total_rows * zero_threshold]
        cols_to_drop = set(sparse_columns).union(zero_columns, low_variance_columns)

        print(f"Dropping {len(sparse_columns)} sparse columns, {len(zero_columns)} all-zero columns, and {len(low_variance_columns)} low-variance columns:")
        print(f"Dropped Columns: {list(cols_to_drop)}")

        self.data.drop(columns=cols_to_drop, inplace=True)

    def imputer(self, knn_threshold=100):

        print("\n Starting Hybrid Imputation (KNN + Interpolation)...")

        nan_columns = self.data.columns[self.data.isna().sum() > 0]

        if nan_columns.empty:
            return

        knn_imputer = KNNImputer(n_neighbors=5)  # KNN Imputer instance

        for column in nan_columns:
            missing_count = self.data[column].isna().sum()
            print(f"\n Imputing column: {column} | Missing: {missing_count}")

            if missing_count < knn_threshold:
                self.data[[column]] = knn_imputer.fit_transform(self.data[[column]])

            else:
                self.interpolate(column)

    def interpolate(self, column):
        
        self.data[column] = self.data[column].interpolate(method="linear", limit_direction="both")
        remaining_nans = self.data[column].isna().sum()
        if remaining_nans > 0:
            self.data[column].fillna(self.data[column].mean(), inplace=True)

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
    





