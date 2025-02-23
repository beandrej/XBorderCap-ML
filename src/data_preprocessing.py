import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../01_data/01_feature_variables/entsoe/demand')

def load_feature_data(country):
    
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../01_data/01_feature_variables')
    data = {}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == f'{country}.csv':
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, parse_dates=True, index_col=0, date_format='%Y-%m-%d %H:%M:%S')
                feature_type = os.path.basename(root)
                data[feature_type] = df

    if data:
        combined_df = pd.concat(data.values(), axis=1)
        return combined_df
    else:
        raise FileNotFoundError(f"No data found for country '{country}' in {base_path}")



aus = load_feature_data('AUS')
bel = load_feature_data('BEL')


print(aus.info())
print(aus.head())
print(bel.info())
print(bel.head())

