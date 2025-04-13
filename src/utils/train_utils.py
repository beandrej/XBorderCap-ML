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
from utils.plot_utils_old import plotLagCorr
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
    
"""************************************************************************
            Training and preprocessing functions
************************************************************************"""

def preparePaths(training_set, model_name, border):
    border_type = training_set.split('_')[1]
    base_path = config.PROJECT_ROOT

    if model_name == 'LSTM':
        model_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}/SEQ_LEN={config.SEQ_LEN}')
        metrics_base = os.path.join(base_path, f'src/results/model_metrics/{border_type}/{model_name}/SEQ_LEN={config.SEQ_LEN}')

        model_path = os.path.join(model_base, f"{model_name}_{training_set}_{border}.pth")
        metrics_path = os.path.join(metrics_base, f"metrics_{model_name}_{training_set}_{border}.csv")

    elif model_name == 'Hybrid':
        model_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}')
        model_config_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}/model_config')
        classmapping_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}/mappings')  
        metrics_base = os.path.join(base_path, f'src/results/model_metrics/{border_type}/{model_name}')  

        model_path = os.path.join(model_base, f'{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.pth')
        model_config_path = os.path.join(model_config_base, f'params_{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.json')
        classmapping_path = os.path.join(classmapping_base, f'cls_map_{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.json')
        metrics_path = os.path.join(metrics_base, f"metrics_{model_name}_{training_set}_{border}_{config.UNIQUE_VAL_TRSH}.csv")
        os.makedirs(model_config_base, exist_ok=True)
        os.makedirs(classmapping_base, exist_ok=True)

    else:
        model_base = os.path.join(base_path, f'model_params/{border_type}/{model_name}')
        metrics_base = os.path.join(base_path, f'src/results/model_metrics/{border_type}/{model_name}')

        model_path = os.path.join(model_base, f"{model_name}_{training_set}_{border}.pth")
        metrics_path = os.path.join(metrics_base, f"metrics_{model_name}_{training_set}_{border}.csv")

    os.makedirs(model_base, exist_ok=True)
    os.makedirs(metrics_base, exist_ok=True)

    if model_name == 'Hybrid':
        return model_path, model_config_path, classmapping_path, metrics_path
    else:
        return model_path, metrics_path

def buildTrainValTestSet(dataset, border):
    
    df = loadDataset(dataset)
    border_type = dataset.split('_')[1]
    X, Y = splitXY(df, border_type)

    time_cols =  ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos"]
    X_time = X[time_cols]
    X_wo_time = X.drop(columns=time_cols)

    X_wo_time = addRollingFeatures(X_wo_time, config.ROLLING_HOURS)

    if not config.PREDICT_ALL_BORDERS and border is not None:
        Y = Y[[border]]
        neighbors = extractCountryNeighbors([
                "AUS_CZE", "CZE_AUS", "AUS_GER", "GER_AUS", "BEL_FRA", "FRA_BEL",
                "BEL_GER", "GER_BEL", "BEL_NET", "NET_BEL", "CZE_GER", "GER_CZE",
                "CZE_POL", "POL_CZE", "GER_NET", "NET_GER", "GER_POL", "POL_GER",
                "GER_FRA", "FRA_GER", "AUS_ITA", "ITA_AUS", "AUS_SWI", "SWI_AUS",
                "BEL_GBR", "GBR_BEL", "SVN_ITA", "ITA_SVN", "DK1_GER", "GER_DK1",
                "DK1_NET", "NET_DK1", "DK2_GER", "GER_DK2", "ESP_FRA", "FRA_ESP",
                "ESP_POR", "POR_ESP", "FRA_GBR", "GBR_FRA", "FRA_ITA", "ITA_FRA",
                "FRA_SWI", "SWI_FRA", "GBR_NET", "NET_GBR", "GER_SWI", "SWI_GER",
                "ITA_SWI", "SWI_ITA", "NET_NO2", "NO2_NET"
            ])
        X, related_countries = filterByBorder(X_wo_time, border, X_time, neighbors)
        print(f"Using {len(X.columns)} input columns for countries: {related_countries}")

        if config.PLOT_LAG_CORR:
            plotLagCorr(pd.concat([X, Y], axis=1), target_col=border)

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
    path = os.path.join(config.PROJECT_ROOT, 'prep_data', f"{dataset_name}.csv")
    df = pd.read_csv(path, index_col=0)
    return df

def filterByBorder(X, border, X_time, country_neighbors):

    countries = border.split("_")
    related = set(countries)
    for c in countries:
        related.update(country_neighbors.get(c, []))

    feature_mask = lambda col: any(cc in col for cc in related)
    selected = [col for col in X.columns if feature_mask(col) or '_rollavg_' in col and feature_mask(col)]

    X_filtered = X[selected]
    X_time = X_time.reindex(X_filtered.index)
    return pd.concat([X_filtered, X_time], axis=1), sorted(related)

def extractCountryNeighbors(target_columns):
    country_neighbors = defaultdict(set)
    for pair in target_columns:
        if "_" in pair:
            c1, c2 = pair.split("_")
            country_neighbors[c1].add(c2)
            country_neighbors[c2].add(c1)
    return {country: sorted(list(neighbors)) for country, neighbors in country_neighbors.items()}

def addRollingFeatures(X: pd.DataFrame, windows: list) -> pd.DataFrame:
    rolling_features_list = []
    for window in windows:
        rolling_avg = X.rolling(window=window, min_periods=1).mean()
        rolling_avg.columns = [f"{col}_rollavg_{window}h" for col in rolling_avg.columns]
        rolling_features_list.append(rolling_avg)
    return pd.concat([X] + rolling_features_list, axis=1)

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
        train_dataset = SequenceDataset(X_train, Y_train, seq_len=config.SEQ_LEN, min_seq_len=24)
        val_dataset = SequenceDataset(X_val, Y_val, seq_len=config.SEQ_LEN, min_seq_len=config.SEQ_LEN)
        collate_fn = padCollate
    else:
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(Y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(Y_val, dtype=torch.float32))
        collate_fn = None

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

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

    X_train, X_val, X_test, pca = preprocessFeatures(X_train, X_val, X_test)

    if config.DO_PREDICT and not config.DO_TRAIN:
        return X_test, Y_test

    if config.PLOT_BORDER_SPLIT:
        return None

    Y_train = Y_train.to_numpy()
    Y_val = Y_val.to_numpy()

    return X_train, Y_train, X_val, Y_val, pca

def prepareDataHybrid(dataset, border):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = buildTrainValTestSet(dataset, border)

    if config.USE_RF:
        RFAnalysis(X_train, Y_train, dataset, border, original_feature_names=X_train.columns.tolist())
        return None
    
    X_train, X_val, X_test, pca = preprocessFeatures(X_train, X_val, X_test)

    if config.DO_PREDICT and not config.DO_TRAIN:
        return X_test, Y_test
    
    _, _, classmap_path, _ = preparePaths(dataset, 'Hybrid', border)
    target_types = getTargetTypes(Y_train, config.UNIQUE_VAL_TRSH)
    cls_cols = [border] if target_types[border] == 'classification' else []
    reg_cols = [] if cls_cols else [border]

    label_encoders = {}
    Y_cls_train, Y_cls_val = pd.DataFrame(), pd.DataFrame()
    full_Y = pd.concat([Y_train, Y_val], axis=0)

    for col in cls_cols:
        full_col = full_Y[col]
        saveClassMappings(Y_train, [col], classmap_path)
        train_encoded, val_encoded, le = safeLabelEncode(Y_train[col], Y_val[col], fit_on=None)
        Y_cls_train[col] = train_encoded
        Y_cls_val[col] = val_encoded
        label_encoders[col] = le

    

    Y_reg_train, Y_reg_val = pd.DataFrame(), pd.DataFrame()
    if reg_cols:
        Y_reg_train = Y_train[reg_cols].copy()
        Y_reg_val = Y_val[reg_cols].copy()

    return (
        X_train, Y_cls_train, Y_reg_train,
        X_val, Y_cls_val, Y_reg_val,
        label_encoders, cls_cols, reg_cols, pca
    )

def createDataloadersHybrid(X_train, Y_cls_train, Y_reg_train, 
                             X_val, Y_cls_val, Y_reg_val, batch_size):

    X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    X_tensor_val = torch.tensor(X_val, dtype=torch.float32)

    Y_cls_tensors_train = [torch.tensor(Y_cls_train[col].values, dtype=torch.long) for col in Y_cls_train.columns]
    Y_cls_tensors_val = [torch.tensor(Y_cls_val[col].values, dtype=torch.long) for col in Y_cls_val.columns]

    Y_reg_tensors_train = [torch.tensor(Y_reg_train[col].values, dtype=torch.float32).view(-1, 1) for col in Y_reg_train.columns]
    Y_reg_tensors_val = [torch.tensor(Y_reg_val[col].values, dtype=torch.float32).view(-1, 1) for col in Y_reg_val.columns]

    train_dataset = HybridDataset(X_tensor_train, Y_cls_tensors_train, Y_reg_tensors_train)
    val_dataset = HybridDataset(X_tensor_val, Y_cls_tensors_val, Y_reg_tensors_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_tensor_train.shape[1]

def preprocessFeatures(X_train, X_val, X_test):

    X_scaler = config.SCALER()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)

    if config.USE_PCA:
        pca = PCA(n_components=config.PCA_COMP)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        return X_train_pca, X_val_pca, X_test_pca, pca
    else:
        return X_train_scaled, X_val_scaled, X_test_scaled, None


"""************************************************************************
            Hybrid model functions
************************************************************************"""

def saveClassMappings(Y_train, cls_cols, path):
    class_mappings = {}

    for col in cls_cols:
        le = LabelEncoder()
        le.fit(Y_train[col])
        mapping = {int(i): str(label) for i, label in enumerate(le.classes_)}
        class_mappings[col] = mapping

    with open(path, "w") as f:
        json.dump(class_mappings, f, indent=2)

    print("✅ Saved class mappings to:", path)

def safeLabelEncode(train_col, val_col, fit_on=None, use_fit_on=config.CLASSIFY_WHOLE_DATASET):
    le = LabelEncoder()
    
    if use_fit_on and fit_on is not None:
        le.fit(fit_on)
    else:
        le.fit(train_col)

    train_encoded = le.transform(train_col)

    class_map = {label: i for i, label in enumerate(le.classes_)}
    val_encoded = val_col.map(class_map).fillna(-1).astype(int)

    return train_encoded, val_encoded, le

def getTargetTypes(Y_train, threshold):
    target_types = {}
    for col in Y_train.columns:
        unique_count = Y_train[col].nunique()
        if unique_count > threshold:
            target_types[col] = 'regression'
        else:
            target_types[col] = 'classification'
    return target_types

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

    if model_name == 'LSTM':
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
    if model_name == "LSTM":
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

