import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import config

class CrossBorderData(Dataset):

    def __init__(self, train=True):

        country1 = 'AUS'
        country2 = 'SWI'
        domain = 'ntc'
        
        border, start, end = self.read_target_variable(domain, country1, country2)

        aus = self.read_feature_data(country1, start, end)
        swi = self.read_feature_data(country2, start, end)
        
        # check for missing timestamps
        border, aus = self.fix_missing_timestamps(border, aus)

        # put together country data & border  
        train_df, test_df = self.put_together(swi, aus, border)

        self.data = train_df if train else test_df  # Load train or test set

        # convert to tensors
        self.X = torch.tensor(self.data.drop(columns=["cross_border_capacity"]).values, dtype=torch.float32)
        self.y = torch.tensor(self.data["cross_border_capacity"].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def read_feature_data(self, country, start, stop):
    
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables')
        data = {}

        # Read all CSV files from a single country into one DF
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file == f'{country}.csv':
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, parse_dates=True, index_col=0, date_format='%Y-%m-%d %H:%M:%S')
                    feature_type = os.path.basename(root)
                    data[feature_type] = df


        combined_df = pd.concat(data.values(), axis=1)
        combined_df = self.cleaned_feature_data(combined_df, start, stop)
        combined_df = self.add_installed_cap(combined_df, country)
        combined_df = combined_df.add_prefix(f"{country}_")

        return combined_df
    
    def add_installed_cap(self, hourly_df, country):
        
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables/entsoe/installed_capacity', f"{country}_installed_generation_capacity.csv")
        
        df = pd.read_csv(path, index_col=1)
        df = df.drop(columns=[df.columns[0]])
        df.index = pd.to_datetime(df.index) 
        df['year'] = df.index.year
        df = df.set_index('year')

        hourly_df['year'] = hourly_df.index.year
        merged_df = hourly_df.merge(df, on='year', how='left')
        merged_df = merged_df.set_index(hourly_df.index)
        merged_df = merged_df.drop(columns=['year'])

        return merged_df
    
    def cleaned_feature_data(self, df, start, stop, mode='cut'):

        start = pd.to_datetime(start)
        stop = pd.to_datetime(stop)
        df.index = pd.to_datetime(df.index)

        og_start, og_stop = df.index.min(), df.index.max()

        df = df.dropna(thresh=50, axis=1)
        if mode == 'cut':
            df = df.loc[start:stop]
        df = df.interpolate(method='linear', limit_direction='both')

        return df

    def read_target_variable(self, domain, country1, country2):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/02_target_variables', f"{domain}", f"{country1}" '_to_' f"{country2}.csv")
        df = pd.read_csv(path, index_col=1)    
        df = df.drop(columns=[df.columns[0]]) 
        df.index = pd.to_datetime(df.index)
        df.columns = ["cross_border_capacity"]
        start, stop = df.index.min(), df.index.max()

        return df, start, stop
    
    def fix_missing_timestamps(self, problematic_df, correct_df):
        full_index = correct_df.index
        problematic_df = problematic_df.reindex(full_index)
        problematic_df = problematic_df.interpolate(method='linear', limit_direction='both')
        print("Fixed missing timestamps and interpolated values!")

        return problematic_df, correct_df

    def print_stats(self):

        print("Starting Timestamp:", f"{self.data.index.min()}")
        print("Ending Timestamp:", f"{self.data.index.max()}")
        print(self.data.info())
        print(self.data.head())



    def check_timestamps(self, df1, df2, mode='check'):
        print("DF1 Time Range:", df1.index.min(), "to", df1.index.max())
        print("DF2 Time Range:", df2.index.min(), "to", df2.index.max())  
        print("DF1 Entries:", len(df1))
        print("DF2 Entries:", len(df2))

        hole1 = df1.index.difference(df2.index)
        hole2 = df2.index.difference(df1.index)
        dups1 = df1.index[df1.index.duplicated()]
        dups2 = df2.index[df2.index.duplicated()]

        problematic_df1 = None
        problematic_df2 = None

        if not hole1.empty:
            print("timestamps NOT IN df2 are TOTAL", len(hole1))
            problematic_df2 = df2    

        if not hole2.empty:
            print("timestamps NOT IN df1 are TOTAL", len(hole2))
            problematic_df1 = df1

        if not dups1.empty:
            print("duplicate timestamps in df1:", dups1.tolist())
            problematic_df1 = df1

        if not dups2.empty:
            print("duplicate timestamps in df2:", dups2.tolist())
            problematic_df2 = df2

        
        if problematic_df1 is not None and problematic_df2 is not None:
            print("both dfs corrupted -> special case TODO")
            raise ValueError("TODO")

        elif problematic_df1 is None and problematic_df2 is None:
            print("Nothing to fix :)")
            return df1, df2  

        # **Fix Missing Timestamps**
        if problematic_df1 is not None:
            print("fixing missing timestamps in df1...")
            df1 = self.fix_missing_timestamps(df1, df2)

        if problematic_df2 is not None:
            print("fixing missing timestamps in df2...")
            df2 = self.fix_missing_timestamps(df2, df1)

        return df1, df2  
    
    def put_together(self, df1, df2, df3):
        assert len(df1) == len(df2) == len(df3)

        df_combined = df1.merge(df2, on="timestamp", how="outer")
        df_combined = df_combined.merge(df3, on="timestamp", how="outer")

        split_index = int(len(df_combined) * config.TRAIN_SPLIT)
        sorted_timestamps = df_combined.index.sort_values()
        split_timestamp = sorted_timestamps[split_index]

        train_df = df_combined.loc[:split_timestamp]
        test_df = df_combined.loc[split_timestamp:]


        return train_df, test_df
    

# country1 = 'AUS'
# country2 = 'SWI'
# domain = 'ntc'

# aus_swi, start, end = read_target_variable(domain, country1, country2)
# aus = read_feature_data(country1, start, end)
# swi = read_feature_data(country2, start, end)

# check_timestamps(aus_swi, aus)
# check_timestamps(aus_swi, swi)
# aus_swi, aus = fix_missing_timestamps(aus_swi, aus)
# check_timestamps(aus_swi, swi)
# check_timestamps(aus_swi, aus)

# train, test = put_together(swi, aus, aus_swi)

# print_stats(train)
# print_stats(test)





