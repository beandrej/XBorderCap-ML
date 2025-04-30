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
from utils.plot_utils import plotLagCorr
import config
from model import getModel

"""************************************************************************
            Dataset classes for training and testing
************************************************************************"""

class SequenceDataset(Dataset):
    def __init__(self, X, Y, seq_len, min_seq_len=None):
        assert len(X) == len(Y)
        self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        self.Y = torch.tensor(Y, dtype=torch.float32) if not torch.is_tensor(Y) else Y
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len or seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        seq_len = self.seq_len
        if self.min_seq_len < self.seq_len:
            seq_len = random.randint(self.min_seq_len, self.seq_len)

        x_seq = self.X[idx : idx + seq_len]
        y_target = self.Y[idx + seq_len]
        return x_seq, y_target, x_seq.shape[0]

class HybridDataset(Dataset):
    def __init__(self, X, Y_cls_list, Y_reg_list):
        self.X = X
        self.Y_cls_list = Y_cls_list
        self.Y_reg_list = Y_reg_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        cls_targets = [y[idx] for y in self.Y_cls_list]
        reg_targets = [y[idx] for y in self.Y_reg_list]
        return self.X[idx], cls_targets, reg_targets
    
class TCNDataset(Dataset):
    def __init__(self, X, Y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32 if Y.ndim == 2 else torch.long)
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
    border_type = training_set.split('_')[1]
    base_path = config.PROJECT_ROOT

    if model_name == 'TCN':
        model_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}')
        metrics_base = os.path.join(base_path, f'src/results/model_metrics/{border_type}/{model_name}')

        model_path = os.path.join(model_base, f"{model_name}_{training_set}_{border}.pth")
        metrics_path = os.path.join(metrics_base, f"metrics_{model_name}_{training_set}_{border}.csv")

    elif model_name == 'Hybrid':
        model_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}')
        model_config_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}/model_config')
        metrics_base = os.path.join(base_path, f'src/results/model_metrics/{border_type}/{model_name}')  

        model_path = os.path.join(model_base, f'{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.pth')
        model_config_path = os.path.join(model_config_base, f'params_{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.json')
        metrics_path = os.path.join(metrics_base, f"metrics_{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.csv")
        os.makedirs(model_config_base, exist_ok=True)

    else:
        model_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}')
        metrics_base = os.path.join(base_path, f'src/results/model_metrics/{border_type}/{model_name}')

        model_path = os.path.join(model_base, f"{model_name}_{training_set}_{border}.pth")
        metrics_path = os.path.join(metrics_base, f"metrics_{model_name}_{training_set}_{border}.csv")

    os.makedirs(model_base, exist_ok=True)
    os.makedirs(metrics_base, exist_ok=True)

    if model_name == 'Hybrid':
        return model_path, metrics_path
    else:
        return model_path, metrics_path

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

    X_train, Y_train, X_val, Y_val, X_test, Y_test = trainValTestSplit(X, Y, config.TRAIN_SPLIT, config.VALID_SPLIT)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

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

def prepareModel(model_name, X_train, Y_train, X_val, Y_val):
    if model_name == "LSTM":
        train_dataset = SequenceDataset(X_train, Y_train, seq_len=config.SEQ_LEN, min_seq_len=None)
        val_dataset = SequenceDataset(X_val, Y_val, seq_len=config.SEQ_LEN, min_seq_len=config.SEQ_LEN)
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
    model = getModel(model_name, input_dim, output_dim).to(config.DEVICE)

    return model, train_loader, val_loader   

def prepareData(training_set, border):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = buildTrainValTestSet(training_set, border)

    if config.USE_RF:
        RFAnalysis(X_train, Y_train, training_set, border, original_feature_names=X_train.columns.tolist())
        return None

    X_train, X_val, X_test = scaleFeatures(X_train, X_val, X_test)

    if config.DO_PREDICT and not config.DO_TRAIN:
        return X_test, Y_test

    if config.PLOT_BORDER_SPLIT:
        return None

    Y_train = Y_train.to_numpy()
    Y_val = Y_val.to_numpy()

    return X_train, Y_train, X_val, Y_val

def prepareDataHybrid(dataset, border):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = buildTrainValTestSet(dataset, border)

    if config.USE_RF:
        RFAnalysis(X_train, Y_train, dataset, border, original_feature_names=X_train.columns.tolist())
        return None

    X_train, X_val, X_test = scaleFeatures(X_train, X_val, X_test)

    # Inference-only mode
    if config.DO_PREDICT and not config.DO_TRAIN:
        if border in config.CLS_COLS:
            classmap_path = os.path.join(config.PROJECT_ROOT, "mappings", "clsMap.json")
            with open(classmap_path, 'r') as f:
                class_mapping = json.load(f)
            mapping = class_mapping[border]
            encoded_test = Y_test[border].astype(str).map(mapping).astype(int)
            Y_test = pd.DataFrame({border: encoded_test})
        return X_test, Y_test

    # Training mode
    if border in config.CLS_COLS:
        classmap_path = os.path.join(config.PROJECT_ROOT, "mappings", "clsMap.json")
        with open(classmap_path, 'r') as f:
            class_mapping = json.load(f)
        mapping = class_mapping[border]

        Y_train = pd.DataFrame({border: Y_train[border].astype(str).map(mapping).astype(int)})
        Y_val = pd.DataFrame({border: Y_val[border].astype(str).map(mapping).astype(int)})

    elif border in config.REG_COLS:
        Y_train = Y_train[[border]]
        Y_val = Y_val[[border]]

    else:
        raise ValueError(f"Border '{border}' not found in CLS_COLS or REG_COLS.")

    return X_train, Y_train, X_val, Y_val

def createTCNDataloaders(X_train, Y_cls_train, Y_reg_train, 
                         X_val, Y_cls_val, Y_reg_val, batch_size, seq_len):

    if not Y_cls_train.empty:
        col = Y_cls_train.columns[0]
        Y_train = Y_cls_train[col].values
        Y_val = Y_cls_val[col].values
    else:
        col = Y_reg_train.columns[0]
        Y_train = Y_reg_train[col].values
        Y_val = Y_reg_val[col].values

    train_dataset = TCNDataset(X_train, Y_train, seq_len)
    val_dataset = TCNDataset(X_val, Y_val, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=True)

    input_dim = X_train.shape[1]
    return train_loader, val_loader, input_dim

def createDataloadersHybrid(X_train, Y_train, X_val, Y_val, task_type):
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

    train_dataset = TensorDataset(X_tensor_train, Y_tensor_train)
    val_dataset = TensorDataset(X_tensor_val, Y_tensor_val)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, X_tensor_train.shape[1]

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


"""************************************************************************
            Hybrid model functions
************************************************************************"""

def getLoader(X):
    X = torch.tensor(X, dtype=torch.float32)
    X = DataLoader(TensorDataset(X), batch_size=config.BATCH_SIZE, shuffle=False)
    return X

def loadHybridClassMap(target_col, mapping_path):
    if not os.path.exists(mapping_path):
        return {}
    with open(mapping_path, "r") as f:
        class_mappings = json.load(f)
    return class_mappings.get(target_col, {})

def loadHybridModelConfig(config_path):
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r") as f:
        return json.load(f)

def computeGlobalMAE(reg_preds, reg_trues, cls_preds, cls_trues):
    mae_list = []

    # Regression heads
    if len(reg_preds) > 0:
        for i in range(reg_preds.shape[1]):
            mae = mean_absolute_error(reg_trues[:, i], reg_preds[:, i])
            mae_list.append(mae)

    # Classification heads
    for cp, ct in zip(cls_preds, cls_trues):
        cp = np.array(cp)
        ct = np.array(ct)

        # Sanity check
        if len(cp) == 0 or len(ct) == 0:
            continue

        min_len = min(len(cp), len(ct))
        cp = cp[:min_len]
        ct = ct[:min_len]

        mae = mean_absolute_error(ct, cp)
        mae_list.append(mae)

    return np.mean(mae_list) if mae_list else 0.0


"""************************************************************************
            Prediction functions
************************************************************************"""

def prepareTestPaths(training_set, model_name, border):
    base_path = config.PROJECT_ROOT
    border_type = training_set.split('_')[1]

    if model_name == 'V2':
        model_path = os.path.join(base_path, f'model_params/{border_type}/{model_name}/SEQ_LEN={config.SEQ_LEN}', f"{model_name}_{training_set}_{border}.pth")
        pred_path = os.path.join(base_path, f'src/results/predictions_csv/{model_name}/SEQ_LEN={config.SEQ_LEN}', f"pred_{model_name}_{training_set}.csv")
        metrics_path = os.path.join(base_path, f'src/results/test_metrics/{model_name}/SEQ_LEN={config.SEQ_LEN}', f"test_metrics_{model_name}_{training_set}.csv")

    elif model_name == 'Hybrid':
        model_path = os.path.join(base_path, f'model_params/{border_type}/{model_name}', f"{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.pth")
        pred_path = os.path.join(base_path, f'src/results/predictions_csv/{model_name}', f"pred_{model_name}_{training_set}.csv")
        metrics_path = os.path.join(base_path, f'src/results/test_metrics/{model_name}', f"test_metrics_{model_name}_{training_set}.csv")
        classmapping_path = os.path.join(base_path, f'model_params/{border_type}/{model_name}/mappings', f'cls_map_{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.json')
        model_config_path = os.path.join(base_path, f'model_params/{border_type}/{model_name}/model_config', f'params_{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.json')
        os.makedirs(os.path.dirname(classmapping_path), exist_ok=True)
        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
    else:
        model_path = os.path.join(base_path, f'model_params/{border_type}/{model_name}', f"{model_name}_{training_set}_{border}.pth")
        pred_path = os.path.join(base_path, f'src/results/predictions_csv/{model_name}', f"pred_{model_name}_{training_set}.csv")
        metrics_path = os.path.join(base_path, f'src/results/test_metrics/{model_name}', f"test_metrics_{model_name}_{training_set}.csv")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    if model_name == 'Hybrid':
        return model_path, pred_path, metrics_path, classmapping_path, model_config_path
    else:
        return model_path, pred_path, metrics_path

def preprocessTestData(X_test):
    X_scaler = config.SCALER()
    X_scaled = X_scaler.fit_transform(X_test)

    if config.USE_PCA:
        pca = PCA(n_components=config.PCA_COMP)
        X_scaled = pca.fit_transform(X_scaled)

    return X_scaled

def prepareTestLoader(X_test, model_name):
    if model_name == "V2":
        dummy_Y = np.zeros(len(X_test))
        dataset = SequenceDataset(X_test, dummy_Y, seq_len=config.SEQ_LEN, min_seq_len=config.SEQ_LEN)
        collate_fn = padCollate
        sample_X, _, _ = dataset[0]
        input_dim = sample_X.shape[1]
    else:
        if isinstance(X_test, pd.DataFrame):
            tensor = torch.tensor(X_test.values, dtype=torch.float32)
        else:
            tensor = torch.tensor(X_test, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        collate_fn = None
        input_dim = tensor.shape[1]

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return loader, input_dim


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

