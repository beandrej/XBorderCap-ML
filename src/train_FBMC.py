import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from model import get_model
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA


"""
************************************
    CONFIG
************************************
"""
LOOP_TRAINING = True
BORDER_TYPE = "MAXBEX"

DATASET_LOOP = ["BL_FBMC_FULL"]
MODEL_LOOP = ["LSTM"]
CRITERIA_LOOP = [
    (nn.MSELoss, "MSELoss")
]
SCALER = StandardScaler

TRAIN_SPLIT = 0.90
VALID_SPLIT = 0.20
BATCH_SIZE = 128
EPOCHS = 100
WEIGHT_DECAY = 3e-3
LEARNING_RATE = 3e-4
SEED = 42
PCA_COMP = 128
SEQ_LEN = 8

USE_PCA = True
USE_RF_ONLY = False
MAKE_PLOTS = True
SHOW_PLOTS = False

"""
************************************
    HELPER FUNC
************************************
"""

#TODO Scale targets separately

class SequenceDataset(Dataset):
    def __init__(self, X, Y, seq_len):
        assert len(X) == len(Y)
        self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        self.Y = torch.tensor(Y, dtype=torch.float32) if not torch.is_tensor(Y) else Y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]               
        y_target = self.Y[idx + self.seq_len]                  
        return x_seq, y_target

def print_nan_summary(df):
    total_nans = df.isna().sum().sum()
    print(f"Total NaNs in dataset: {total_nans}")

    if total_nans > 0:
        print("\n🔍 NaNs per column:")
        print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))

def RFAnalysis(TRAINING_SET):
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{TRAINING_SET}.csv"), index_col=0)
    print(f"\n USING DATASET: {TRAINING_SET}\n")

    # Dataset X & Y has to merged (only use intersecting timestamps), they are separated again here..
    if BORDER_TYPE == "MAXBEX":
        first_target_idx = df.columns.get_loc("AUS_CZE")
    elif BORDER_TYPE == "NTC":
        first_target_idx = df.columns.get_loc("AT_to_IT_NORD")
    else:
        raise ValueError("Wrong BORDER_TYPE!")

    X = df.iloc[:, :first_target_idx]
    Y = df.iloc[:, first_target_idx:]

    split_index = int(len(df) * TRAIN_SPLIT)
    val_index = int(split_index * (1 - VALID_SPLIT))

    X_train_full = X.iloc[:split_index]
    Y_train_full = Y.iloc[:split_index]

    X_train = X_train_full.iloc[:val_index]
    Y_train = Y_train_full.iloc[:val_index]

    feature_names = X.columns.tolist()
    print("\nRunning Random Forest feature importance analysis...\n")
    feature_importance_df = analyze_feature_importance(X_train, Y_train, feature_names, TRAINING_SET)
    feature_importance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/dataset_metrics/RF_{TRAINING_SET}_feature_weights.csv')
    os.makedirs(os.path.dirname(feature_importance_path), exist_ok=True)
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print(f"Feature importance saved to: {feature_importance_path}\n")
    exit() 

def analyze_feature_importance(X_train, Y_train, feature_names, dataset, top_n=40, saveFig=True, showPlot=False):
    rf = ExtraTreesRegressor(
        n_estimators=1000,     
        max_depth=None,         
        min_samples_split=5, 
        max_features='sqrt',  
        n_jobs=-1,            
        random_state=SEED     
    )
    rf.fit(X_train, Y_train)
    importances = rf.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("Top features:\n", feature_importance_df.head(top_n))

    plt.figure(figsize=(12, 6))
    top_features = feature_importance_df.head(top_n)
    plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
    plt.xlabel('Importance')
    plt.title('Top Feature Importances from Random Forest')
    plt.tight_layout()
    if saveFig: plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/plots/dataset_metrics_RF', f"RF_{dataset}_weights.png"))
    if showPlot: plt.show()
        

    return feature_importance_df

def plotTrainValLoss(metrics_df, loss, dataset, model, saveFig=True, showPlot=False):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Training Loss", color="blue")
    ax.plot(metrics_df["epoch"], metrics_df["val_loss"], label="Validation Loss", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss {loss}")
    ax.set_title(f"{loss} of Training vs Validation loss of {model} on {dataset}")
    ax.legend()
    ax.grid()
    if saveFig: plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/plots/training/losses', f"{model}_{dataset}_{loss}_TrainValLoss.png"), dpi=300, bbox_inches='tight') 
    if showPlot: plt.show()

def plotR2(metrics_df, dataset, model, loss, saveFig=True, showPlot=False):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(metrics_df["epoch"], metrics_df["train_r2"], label="Training R2-Score", color="blue")
    ax.plot(metrics_df["epoch"], metrics_df["val_r2"], label="Validation R2-Score", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2-Score")
    ax.set_title(f"Training and Validation R2-Score of {model} on {dataset}")
    ax.legend()
    ax.grid()
    if saveFig: plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/plots/training/r2', f"{model}_{dataset}_{loss}_R2Score.png"), dpi=300, bbox_inches='tight')
    if showPlot: plt.show()     

def buildTrainValDataset(dataset, border):
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{dataset}.csv"), index_col=0)
    print(f"\n USING DATASET: {dataset}\n")
    #print_nan_summary(df)

    # Dataset X & Y has to merged (only use intersecting timestamps), they are separated again here..
    if BORDER_TYPE == "MAXBEX":
        first_target_idx = df.columns.get_loc("AUS_CZE")
    elif BORDER_TYPE == "NTC":
        first_target_idx = df.columns.get_loc("AT_to_IT_NORD")
    else:
        raise ValueError("Wrong BORDER_TYPE!")

    X = df.iloc[:, :first_target_idx]
    Y = df.iloc[:, first_target_idx:]
    if border is not None:
        Y = Y[[border]]

    split_index = int(len(df) * TRAIN_SPLIT)
    val_index = int(split_index * (1 - VALID_SPLIT))

    X_train_full = X.iloc[:split_index]
    Y_train_full = Y.iloc[:split_index]

    X_train = X_train_full.iloc[:val_index]
    Y_train = Y_train_full.iloc[:val_index]

    X_val = X_train_full.iloc[val_index:]
    Y_val = Y_train_full.iloc[val_index:]


    
    return X_train, Y_train, X_val, Y_val

"""
************************************************************************************************************************************************************************************
                                                                    TRAINING LOOP
************************************************************************************************************************************************************************************
"""


def main(TRAINING_SET, MODEL_NAME, CRITERION_CLASS, border):

    if MODEL_NAME == 'LSTM':
        torch_model_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../model_params/{BORDER_TYPE}/{MODEL_NAME}/SEQ_LEN={SEQ_LEN}')
        metrics_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_metrics/{BORDER_TYPE}/{MODEL_NAME}/SEQ_LEN={SEQ_LEN}')
        torch_model_path = os.path.join(torch_model_base_path, f"{MODEL_NAME}_{TRAINING_SET}_{border}_{CRITERION_CLASS().__class__.__name__}_{SEQ_LEN}.pth")
        metrics_path = os.path.join(metrics_base_path, f"metrics_{MODEL_NAME}_{TRAINING_SET}_{border}_{CRITERION_CLASS().__class__.__name__}_{SEQ_LEN}.csv")
    else:
        torch_model_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../model_params/{BORDER_TYPE}/{MODEL_NAME}')
        metrics_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_metrics/{BORDER_TYPE}/{MODEL_NAME}')
        torch_model_path = os.path.join(torch_model_base_path, f"{MODEL_NAME}_{TRAINING_SET}_{border}_{CRITERION_CLASS().__class__.__name__}.pth")
        metrics_path = os.path.join(metrics_base_path, f"metrics_{MODEL_NAME}_{TRAINING_SET}_{border}_{CRITERION_CLASS().__class__.__name__}.csv")

    summaryXLSX_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_metrics', "00_summary.xlsx")

    os.makedirs(torch_model_base_path, exist_ok=True)
    os.makedirs(metrics_base_path, exist_ok=True)
    os.makedirs(os.path.dirname(summaryXLSX_path), exist_ok=True)
    
    if USE_RF_ONLY:
        RFAnalysis(TRAINING_SET)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    X_train, Y_train, X_val, Y_val = buildTrainValDataset(TRAINING_SET, border)

    print("Shape of X_train:", X_train.shape) 
    print("Shape of Y_train:", Y_train.shape) 
    print("Shape of X_val:", X_val.shape) 
    print("Shape of Y_val:", Y_val.shape) 

    if USE_PCA:
        pca = PCA(n_components=PCA_COMP) 
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
    else:
        X_scaler = SCALER()
        X_train = X_scaler.fit_transform(X_train)
        X_val = X_scaler.transform(X_val)

    Y_scaler = SCALER()
    Y_train = Y_scaler.fit_transform(Y_train)
    Y_val = Y_scaler.transform(Y_val)


    if MODEL_NAME == "LSTM":
        train_dataset = SequenceDataset(X_train, Y_train, seq_len=SEQ_LEN)
        val_dataset = SequenceDataset(X_val, Y_val, seq_len=SEQ_LEN)
    else:
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                    torch.tensor(Y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(Y_val, dtype=torch.float32))

    sample_X, sample_Y = train_dataset[0]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if MODEL_NAME == "LSTM":
        input_dim = sample_X.shape[1]  
    else:
        input_dim = sample_X.shape[0] 
    output_dim = sample_Y.shape[0]

    model = get_model(MODEL_NAME, input_dim, output_dim).to(device)
    
    criterion = CRITERION_CLASS()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    train_r2_scores = []
    train_mae_scores = []
    val_r2_scores = []
    val_mae_scores = []
    best_val_r2 = -np.inf 

    if USE_PCA:
        print(f"\n PCA ENABLED: Reduction to {PCA_COMP} dim. before feeding into Train Loop\n")
    else:
        print("\n PCA DISABLED\n")
    if MODEL_NAME == 'LSTM':
        print(f"\nLSTM Model with SEQ_LEN = {SEQ_LEN}\n")
    print(f"\nStarting Training: \n \nTrain Split: {int(TRAIN_SPLIT*100)}% | Val Split: {int(VALID_SPLIT*100)}%")
    print(f"Batch size: {BATCH_SIZE} | Weight Decay: {WEIGHT_DECAY} | Learning Rate: {LEARNING_RATE}")

    """
    ************************************************************************************************************************************************************************************
                                                                        TRAINING LOOP
    ************************************************************************************************************************************************************************************
    """

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0  

        y_train_true_list = []
        y_train_pred_list = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            y_train_pred = predictions.detach().cpu().numpy()
            y_train_true = y_batch.detach().cpu().numpy()

            y_train_pred = Y_scaler.inverse_transform(y_train_pred)
            y_train_true = Y_scaler.inverse_transform(y_train_true)

            y_train_pred_list.append(y_train_pred)
            y_train_true_list.append(y_train_true)

        y_train_pred_full = np.vstack(y_train_pred_list)
        y_train_true_full = np.vstack(y_train_true_list)

        train_r2 = r2_score(y_train_true_full, y_train_pred_full)
        train_mae = mean_absolute_error(y_train_true_full, y_train_pred_full)

        train_r2_scores.append(train_r2)
        train_mae_scores.append(train_mae)


        """
        ************************************************************************************************************************************************************************************
                                                                            VALIDATION
        ************************************************************************************************************************************************************************************
        """


        model.eval()
        val_loss = 0
        y_val_true_list = []
        y_val_pred_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred_log = model(X_batch)
                loss = criterion(y_pred_log, y_batch)
                val_loss += loss.item()

                y_pred = y_pred_log.detach().cpu().numpy()
                y_true = y_batch.detach().cpu().numpy()

                y_pred = Y_scaler.inverse_transform(y_pred)
                y_true = Y_scaler.inverse_transform(y_true)

                y_val_pred_list.append(y_pred)
                y_val_true_list.append(y_true)    


        y_pred_full = np.vstack(y_val_pred_list)
        y_true_full = np.vstack(y_val_true_list)

        val_r2 = r2_score(y_true_full, y_pred_full)
        val_mae = mean_absolute_error(y_true_full, y_pred_full)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), torch_model_path)

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_r2_scores.append(val_r2)
        val_mae_scores.append(val_mae)

        scheduler.step(val_loss)

        if (epoch+1) % 4 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train {criterion.__class__.__name__}: {train_losses[-1]:.2f} | Val {criterion.__class__.__name__}: {val_losses[-1]:.2f} | Train-R²: {train_r2:.2f} | Val-R²: {val_r2:.2f} | Train-MAE: {train_mae:.2f} | Val-MAE: {val_mae:.2f}")


    """
    ************************************************************************************************************************************************************************************
                                                                        METRICS
    ************************************************************************************************************************************************************************************
    """


    epochs_list = list(range(1, EPOCHS + 1))
    metrics_df = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_r2': train_r2_scores,
        'val_r2' : val_r2_scores,
        'train_mae': train_mae_scores,
        'val_mae' : val_mae_scores
        
    })

    if MAKE_PLOTS:
        plotTrainValLoss(metrics_df, criterion.__class__.__name__, TRAINING_SET, MODEL_NAME)
        plotR2(metrics_df, TRAINING_SET, MODEL_NAME, criterion.__class__.__name__)

    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    for loss, loss_name in CRITERIA_LOOP:
        for model_name in MODEL_LOOP:
            for dataset in DATASET_LOOP:
                _, Y_all, _, _ = buildTrainValDataset(dataset, border=None)  # Get all target names
                for target_col in Y_all.columns:
                    print(f"\n====== Training {model_name} on {dataset} | Target: {target_col} | Loss: {loss_name} ======\n")
                    main(dataset, model_name, loss, target_col)
