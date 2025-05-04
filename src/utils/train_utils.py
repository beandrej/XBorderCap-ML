import sys
import os
import torch
import random
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

"""************************************************************************
            Dataset classes for training and testing
************************************************************************"""

class SequentialDataset(Dataset):
    def __init__(self, X, Y, seq_len):
        self.X = X.clone().detach().float() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        if isinstance(Y, torch.Tensor):
            self.Y = Y.clone().detach()
        else:
            self.Y = torch.tensor(Y)
        
        self.Y = self.Y.float() if self.Y.ndim == 2 else self.Y.long()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]  
        y = self.Y[idx + self.seq_len - 1]       
        return x_seq, y

"""************************************************************************
            Training and preprocessing functions
************************************************************************"""

def preparePaths(training_set, model_name, border):
    dataset_type = training_set.split('_')[0]
    border_type = training_set.split('_')[1]
    base_path = config.PROJECT_ROOT

    if config.ENABLE_BACKTEST:
        data_path = os.path.join(base_path, "data_cache/backtest", training_set, f"{dataset_type}_{border}.pt")
        model_path = os.path.join(base_path, f'model_params/BACKTEST/{border_type}/{model_name}', f"{model_name}_{training_set}_{border}.pth")
        train_metrics_path = os.path.join(base_path, f'src/results/model_metrics/BACKTEST/{border_type}/{model_name}', f"metrics_{model_name}_{training_set}_{border}.csv")
        test_metrics_path = os.path.join(base_path, f'src/results/test_metrics/BACKTEST/{model_name}', f"test_metrics_{model_name}_{training_set}.csv")
        pred_path = os.path.join(base_path, f'src/results/predictions_csv/BACKTEST/{model_name}', f"pred_{model_name}_{training_set}.csv")
    else:
        data_path = os.path.join(base_path, "data_cache", training_set, f"{dataset_type}_{border}.pt")
        model_path = os.path.join(base_path, f'model_params/{border_type}/{model_name}', f"{model_name}_{training_set}_{border}.pth")
        train_metrics_path = os.path.join(base_path, f'src/results/model_metrics/{border_type}/{model_name}', f"metrics_{model_name}_{training_set}_{border}.csv")
        test_metrics_path = os.path.join(base_path, f'src/results/test_metrics/{model_name}', f"test_metrics_{model_name}_{training_set}.csv")
        pred_path = os.path.join(base_path, f'src/results/predictions_csv/{model_name}', f"pred_{model_name}_{training_set}.csv")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)

    return data_path, model_path, train_metrics_path, test_metrics_path, pred_path 

def buildTrainValTestSet(dataset, border):
    
    df = loadDataset(dataset)
    border_type = dataset.split('_')[1]
    X, Y = splitXY(df, border_type)

    time_cols =  ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos"]
    X_time = X[time_cols]
    X_wo_time = X.drop(columns=time_cols)

    Y = Y[[border]]
    neighbors = extractCountryNeighbors(config.ALL_BORDERS)
    X, related_countries = filterByBorder(X_wo_time, border, X_time, neighbors)
    print(f"Using {len(X.columns)} input columns for countries: {related_countries}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test = trainValTestSplitBacktest(X, Y)#, config.TRAIN_SPLIT, config.VALID_SPLIT)

    test_timestamps = X_test.index

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, test_timestamps

def splitXY(df: pd.DataFrame, border_type: str):
    if border_type == "FBMC":
        first_target_idx = df.columns.get_loc("AUS_CZE")
    elif border_type == "NTC":
        first_target_idx = df.columns.get_loc("AUS_ITA")
    else:
        raise ValueError("Wrong BORDER_TYPE!")

    return df.iloc[:, :first_target_idx], df.iloc[:, first_target_idx:]

def loadDataset(dataset_name: str) -> pd.DataFrame:
    data_dir = os.path.join(config.PROJECT_ROOT, 'prep_data')
    base_path = os.path.join(data_dir, dataset_name)

    if os.path.exists(f"{base_path}.parquet"):
        return pd.read_parquet(f"{base_path}.parquet")

    elif os.path.exists(f"{base_path}.csv"):
        df = pd.read_csv(f"{base_path}.csv", index_col=0)
        return df.to_parquet(f"{base_path}.parquet")

def filterByBorder(X, border, X_time, country_neighbors):

    countries = border.split("_")
    related = set(countries)
    for c in countries:
        related.update(country_neighbors.get(c, []))

    feature_mask = lambda col: any(cc in col for cc in related)
    selected = [col for col in X.columns if feature_mask(col)]

    X_filtered = X[selected]
    X_time = X_time.reindex(X_filtered.index)
    return pd.concat([X_filtered, X_time], axis=1), sorted(related)

def extractCountryNeighbors(target_columns):
    country_neighbors = defaultdict(set)
    for pair in target_columns:
        assert("_" in pair), "Error: Border must be in format XXX_YYY"
        c1, c2 = pair.split("_")
        country_neighbors[c1].add(c2)
        country_neighbors[c2].add(c1)
    return {country: sorted(list(neighbors)) for country, neighbors in country_neighbors.items()}

def trainValTestSplitBacktest(X, Y):
    test_start = "2023-07-01 00:00:00"
    test_end = "2023-12-31 23:00:00"

    X = X.copy()
    Y = Y.copy()
    X.index = pd.to_datetime(X.index)
    Y.index = pd.to_datetime(Y.index)

    X_test = X.loc[test_start:test_end]
    Y_test = Y.loc[test_start:test_end]

    X_train_full = X.drop(X_test.index)
    Y_train_full = Y.drop(Y_test.index)

    val_size = int(len(X_train_full) * 0.1)
    X_train = X_train_full.iloc[:-val_size]
    Y_train = Y_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    Y_val = Y_train_full.iloc[-val_size:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def trainValTestSplit(X, Y, train_frac, val_frac):
    assert 0 < train_frac < 1
    assert 0 <= val_frac < 1

    total_len = len(X)
    test_size = int(total_len * (1 - train_frac))
    train_val_len = total_len - test_size

    X_test = X.iloc[-test_size:]
    Y_test = Y.iloc[-test_size:]

    X_train_val = X.iloc[:train_val_len]
    Y_train_val = Y.iloc[:train_val_len]

    val_size = int(train_val_len * val_frac)
    train_size = train_val_len - val_size

    X_train = X_train_val.iloc[:train_size]
    Y_train = Y_train_val.iloc[:train_size]

    X_val = X_train_val.iloc[train_size:]
    Y_val = Y_train_val.iloc[train_size:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def getLoaders(model_name, X_train, Y_train, X_val, Y_val):
    if model_name == "LSTM":
        train_dataset = SequentialDataset(X_train, Y_train, seq_len=config.SEQ_LEN)
        val_dataset = SequentialDataset(X_val, Y_val, seq_len=config.SEQ_LEN)
        collate_fn = padCollate
    else:
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(Y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(Y_val, dtype=torch.float32))
        collate_fn = None

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    if model_name == "LSTM":
        sample_X, sample_Y, _ = train_dataset[0]
        input_dim = sample_X.shape[1]
    else:
        sample_X, sample_Y = train_dataset[0]
        input_dim = sample_X.shape[0]

    output_dim = sample_Y.shape[0] if len(sample_Y.shape) > 0 else 1

    return train_loader, val_loader   

def prepareDataHybrid(dataset, border):
    X_train, Y_train, X_val, Y_val, X_test, Y_test, test_timestamps = buildTrainValTestSet(dataset, border)

    if config.USE_RF:
        RFAnalysis(X_train, Y_train, dataset, border, original_feature_names=X_train.columns.tolist())
        return None

    X_train, X_val, X_test = scaleFeatures(X_train, X_val, X_test)

    if border in config.CLS_COLS:
        classmap_path = os.path.join(config.PROJECT_ROOT, "mappings", "clsMap.json")
        with open(classmap_path, 'r') as f:
            class_mapping = json.load(f)
        mapping = class_mapping[border]

        Y_train = pd.DataFrame({border: Y_train[border].astype(str).map(mapping).astype(int)})
        Y_val   = pd.DataFrame({border: Y_val[border].astype(str).map(mapping).astype(int)})
        Y_test  = pd.DataFrame({border: Y_test[border].astype(str).map(mapping).astype(int)})

    elif border in config.REG_COLS:
        Y_train = Y_train[[border]]
        Y_val   = Y_val[[border]]
        Y_test  = Y_test[[border]]

    else:
        raise ValueError(f"Border '{border}' not found in CLS_COLS or REG_COLS.")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, test_timestamps

def getLoadersHybrid(X_train, Y_train, X_val, Y_val, task_type, model):
    X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    X_tensor_val = torch.tensor(X_val, dtype=torch.float32)

    if task_type == 'classification':
        Y_tensor_train = torch.tensor(Y_train.values.squeeze(), dtype=torch.long)
        Y_tensor_val = torch.tensor(Y_val.values.squeeze(), dtype=torch.long)
    elif task_type == 'regression':
        Y_tensor_train = torch.tensor(Y_train.values, dtype=torch.float32)
        Y_tensor_val = torch.tensor(Y_val.values, dtype=torch.float32)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

    if model.sequence:
        print("MODEL USING SEQUENCE DATASET")
        train_dataset = SequentialDataset(X_tensor_train, Y_tensor_train, seq_len=config.SEQ_LEN)
        val_dataset = SequentialDataset(X_tensor_val, Y_tensor_val, seq_len=config.SEQ_LEN)
    else:
        train_dataset = TensorDataset(X_tensor_train, Y_tensor_train)
        val_dataset = TensorDataset(X_tensor_val, Y_tensor_val)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_TRAIN, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=False)

    return train_loader, val_loader

def getTestLoader(X_test, Y_test, model):
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    if model.model_type == "Reg":
        Y_tensor = torch.tensor(Y_test.values, dtype=torch.float32)
    else:
        Y_tensor = torch.tensor(Y_test.values.squeeze(), dtype=torch.long)

    if model.sequence:
        test_dataset = SequentialDataset(X_tensor, Y_tensor, seq_len=config.SEQ_LEN)
    else:
        test_dataset = TensorDataset(X_tensor, Y_tensor)

    loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=False)
    return loader

def scaleFeatures(X_train, X_val, X_test):

    X_scaler = config.SCALER()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)

    if config.USE_PCA:
        pca = PCA(n_components=config.PCA_COMP)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        print(f"PCA {config.PCA_COMP} dims → Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")

    return X_train, X_val, X_test

def savePredictions(pred_path, test_timestamps, trues, preds, border, write=True, overwrite=True):

    min_len = min(len(preds), len(trues), len(test_timestamps))

    new_df = pd.DataFrame({
        "timestamp": test_timestamps[:min_len],
        "true": trues[:min_len].flatten(),
        "pred": preds[:min_len].flatten(),
    })

    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])

    if not write:
        print(f"\n --- Predictions NOT SAVED for {border}! --- \n")

    # If no file found and writing enabled
    elif not os.path.exists(pred_path) and write:
        pred_df = new_df

        pred_df.sort_values("timestamp", inplace=True)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        print(f"\n --- NEW Predictions SAVED for {border}! --- \n")

    # If existing file found predictions assumes to be present -> overwrite column
    elif os.path.exists(pred_path) and write and overwrite:
        cols_to_remove = [f"{border}_true", f"{border}_pred"]
        pred_df = pd.read_csv(pred_path)
        pred_df = pred_df.drop(columns=[col for col in cols_to_remove if col in pred_df.columns])

        min_len = min(len(pred_df), len(preds), len(trues))
        pred_df = pred_df.iloc[:min_len].copy()

        pred_df[f"{border}_pred"] = preds[:min_len].flatten()
        pred_df[f"{border}_true"] = trues[:min_len].flatten()
        print(f"\n --- Predictions SAVED: Overwrite = {config.OVERWRITE_PREDICTIONS} for {border}! --- \n")

        pred_df.sort_values("timestamp", inplace=True)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        pred_df.to_csv(pred_path, index=False)

    # If existing file found but NO overwrite
    elif os.path.exists(pred_path) and write and not overwrite:
        print(f"\n --- EXISTING PREDICTIONS FOUND --- Predictions NOT saved for {border}! --- Activate OVERWRITE in config or check existing file.\n")
    else:
        print(f"\n --- Predictions NOT saved for {border}! --- \n")
        pass

    assert len(preds) == len(trues)



"""************************************************************************
            Misc functions
************************************************************************"""

def setSeed():
    seed = config.SEED
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def printFeatureStats(name, X):
    print(f"{name}:")
    print(f"Mean     : {X.mean().mean():.2f}")
    print(f"Variance : {X.var().mean():.2f}")
    print("-" * 60)

def printNaNSummary(df):
    total_nans = df.isna().sum().sum()
    print(f"Total NaNs in dataset: {total_nans}")

    if total_nans > 0:
        print("\nNaNs per column:")
        print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))

def RFAnalysis(X_train, Y_train, dataset_name, border, original_feature_names=None, saveFig=True, showFig=False, printTopFeatures=False):
    # Use original column names if provided
    if original_feature_names is not None:
        feature_names = original_feature_names
    elif isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
    else:
        # fallback
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # If X_train is not a DataFrame, wrap it so we can use feature names in plotting
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    # Make sure Y_train is a 1D array
    if isinstance(Y_train, pd.DataFrame):
        Y_train = Y_train.values.ravel()
    elif isinstance(Y_train, np.ndarray) and Y_train.ndim > 1:
        Y_train = Y_train.ravel()

    print("\nRunning Random Forest feature importance analysis...\n")

    rf = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=config.SEED
    )
    rf.fit(X_train, Y_train)
    importances = rf.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    if printTopFeatures:
        print("Feature importances:\n", feature_importance_df.head(30))

    # Plotting
    plt.figure(figsize=(14, 6))
    top_features = feature_importance_df.head(30)
    plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
    plt.xlabel('Importance')
    plt.title(f'Top Features for {border} - {dataset_name}')
    plt.tight_layout()

    plot_dir = os.path.join(config.PROJECT_ROOT, 'src/results/plots/dataset_metrics_RF')
    os.makedirs(plot_dir, exist_ok=True)

    if saveFig: plt.savefig(os.path.join(plot_dir, f"RF_{dataset_name}_{border}_weights.png"))
    if showFig: plt.show()
    else: plt.close()

    return feature_importance_df

def padCollate(batch):
    x_batch, y_batch, lengths = zip(*batch)

    x_batch = pad_sequence(x_batch, batch_first=True)     
    y_batch = torch.stack(y_batch)                        
    lengths = torch.tensor(lengths)                       

    return x_batch, y_batch, lengths

def getTimestamps(dataset):
    df = loadDataset(dataset)
    if config.PREDICT_ON_FULL_DATA:
        timestamps = pd.to_datetime(df.index)
    else:
        split_index = int(len(df) * config.TRAIN_SPLIT)
        timestamps = pd.to_datetime(df.index[split_index:])
    return timestamps

def borderCheck(border):
    if border.split('_')[1] == 'FBMC':
        return config.FBMC_BORDERS
    elif border.split('_')[1] == 'NTC':
        return config.NTC_BORDERS