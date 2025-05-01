import os
import torch
import sys
import json
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config 
from utils.train_utils import prepareDataHybrid 

def cacheAllBorders(dataset):
    dataset_type = dataset.split('_')[0]
    border_type = dataset.split('_')[1]
    all_borders = config.CLS_COLS + config.REG_COLS
    cache_dir = os.path.join(config.PROJECT_ROOT, "data_cache/backtest", dataset)
    os.makedirs(cache_dir, exist_ok=True)

    for border in tqdm(all_borders, desc="Caching borders"):
        try:
            print(f"\n🟨 Processing {border}...")

            # Updated to return test set
            X_train, Y_train, X_val, Y_val, X_test, Y_test, test_timestamps = prepareDataHybrid(dataset, border)

            if border in config.CLS_COLS:
                with open(os.path.join(config.PROJECT_ROOT, "mappings", "clsMap.json"), 'r') as f:
                    class_mapping = json.load(f)
                out_dim = len(class_mapping[border])
                task_type = 'classification'
            else:
                out_dim = 1
                task_type = 'regression'

            input_dim = X_train.shape[1]

            cache_path = os.path.join(cache_dir, f"{dataset_type}_{border}.pt")
            if os.path.exists(cache_path):
                print(f"⚠️ Skipping {border} (already cached)")
                continue

            torch.save({
                "X_train": X_train,
                "Y_train": Y_train,
                "X_val": X_val,
                "Y_val": Y_val,
                "X_test": X_test,
                "Y_test": Y_test,
                "task_type": task_type,
                "input_dim": input_dim,
                "out_dim": out_dim,
                "test_timestamp": test_timestamps,
            }, cache_path)

            print(f"✅ Cached: {border} -> {cache_path}")

        except Exception as e:
            print(f"❌ Failed to process {border}: {e}")

if __name__ == "__main__":
    dataset = ['BL_NTC_NORM', 'BL_FBMC_NORM', 'FX_FBMC_NORM', 'FX_NTC_NORM']
    for i in dataset:
        cacheAllBorders(i)