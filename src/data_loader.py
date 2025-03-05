import matplotlib.pyplot as plt
import pandas as pd
from data_class import CrossBorderData, TypeData

DATASET_NAME = 'RAIN'

# X-BorderSet
COUNTRY1 = 'AUS'
COUNTRY2 = 'SWI'
DOMAIN = 'ntc'

# TypeSet
SOURCE = 'weather'
DATATYPE = 'precipitation'

enable_plot = False

def main():

    #set = CrossBorderData(COUNTRY1, COUNTRY2, DOMAIN, DATASET_NAME, load_from_file=False)
    test = TypeData(SOURCE, DATATYPE, DATASET_NAME, load_from_file=True, plot_corr=False)

    if enable_plot:
        print("\n📊 Available Features:")
        print(set.feature_columns[:10])

        plot(set, [
            "BEL_generation_hydro_pumped_storage", 
            "AUS_generation_geothermal"
        ], start_date="2015-01-01", end_date="2026-01-01")

def plot(dataset, columns, start_date=None, end_date=None):

    df = pd.DataFrame(dataset.X.numpy(), columns=dataset.feature_columns)
    df["cross_border_capacity"] = dataset.y.numpy().flatten()
    df.index = pd.to_datetime(dataset.timestamp)

    df = df.sort_index()

    available_cols = [col for col in columns if col in df.columns]
    if not available_cols:
        return
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

if __name__ == "__main__":
    main()
