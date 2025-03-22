import os
import torch
import torch.nn as nn
import openpyxl
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from data_class import *
import data_loader
from model import *
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge


"""
************************************
    CONFIG
************************************
"""
TRAINING_SET = "BASELINE_MAXBEX"

TRAIN_SPLIT = 0.8 
VALID_SPLIT = 0.1
BATCH_SIZE = 512
EPOCHS = 100
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 0.001

MODEL_NAME = "BaseModel"
SPLIT_RATIO = 0.8
VALID_SPLIT = 0.1

"""
************************************
"""

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{TRAINING_SET}.csv"), index_col=0)
    first_target_idx = full_df.columns.get_loc("AUS_CZE")

    X = full_df.iloc[:, :first_target_idx]
    Y = full_df.iloc[:, first_target_idx:]

    split_index = int(len(full_df) * SPLIT_RATIO)
    val_index = int(split_index * (1 - VALID_SPLIT))

    X_train_full = X.iloc[:split_index]
    Y_train_full = Y.iloc[:split_index]

    X_train = X_train_full.iloc[:val_index]
    Y_train = Y_train_full.iloc[:val_index]

    X_val = X_train_full.iloc[val_index:]
    Y_val = Y_train_full.iloc[val_index:]

    print("Shape of X_train:", X_train.shape) 
    print("Shape of Y_train:", Y_train.shape) 
    print("Shape of X_val:", X_val.shape) 
    print("Shape of Y_val:", Y_val.shape) 

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)

    Y_train = Y_scaler.fit_transform(Y_train)
    Y_val = Y_scaler.transform(Y_val)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = next(iter(train_loader))[0].shape[1]
    output_dim = Y_train.shape[1]
    model = get_model(MODEL_NAME, input_dim, output_dim).to(device)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print("Model parameters:\n", f"Train split: {SPLIT_RATIO}\n", f"Validation split: {VALID_SPLIT}\n",
        f"Batch size: {BATCH_SIZE}\n", f"Learning rate: {config.LEARNING_RATE}")


    train_losses = []
    val_losses = []
    r2_scores = []
    mae_scores = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0  
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if MODEL_NAME == "lstm":
                X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)

        model.eval()
        val_loss = 0
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                if MODEL_NAME == "lstm":
                    X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                
                y_pred = y_pred.cpu().numpy()
                y_batch = y_batch.cpu().numpy()

                y_pred = Y_scaler.inverse_transform(y_pred)
                y_true = Y_scaler.inverse_transform(y_batch)

                y_pred_list.append(y_pred)
                y_true_list.append(y_true)    

        y_pred_full = np.vstack(y_pred_list)
        y_true_full = np.vstack(y_true_list)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        r2_scores.append(r2)
        mae_scores.append(mae)

        if epoch % 2 == 0:
            print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss (MSE): {train_losses[-1]:.2f} | Val Loss (MSE): {val_losses[-1]:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}")
        scheduler.step()

    epochs_list = list(range(1, config.EPOCHS + 1))
    metrics_df = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'r2': r2_scores,
        'mae': mae_scores
    })

    summary_data = {
        'ModelType': MODEL_NAME,
        'Dataset': TRAINING_SET,
        'Learning Rate': LEARNING_RATE,
        'Weight Decay': WEIGHT_DECAY,
        'Train Split': TRAIN_SPLIT,
        'Batch size': BATCH_SIZE,
        'Epochs': EPOCHS,
        'Train loss': round(train_losses[-1], 2),
        'Val loss': round(val_losses[-1], 2),
        'R2-Score': round(r2_scores[-1], 2),
        'MAE': round(mae_scores[-1], 2)
    }

    summary_df = pd.DataFrame([summary_data])

    summaryXLSX_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_metrics', 
                            f"00_summary.xlsx")

    os.makedirs(os.path.dirname(summaryXLSX_path), exist_ok=True)

    if os.path.exists(summaryXLSX_path):
        existing_df = pd.read_excel(summaryXLSX_path)
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
    else:
        combined_df = summary_df
    combined_df.to_excel(summaryXLSX_path, index=False)
    print(f"Updated summary saved to: {summaryXLSX_path}")

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_metrics', f"metrics_{MODEL_NAME}_{TRAINING_SET}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")

    torch_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_params', f"{MODEL_NAME}.pth")
    torch.save(model.state_dict(), torch_model_path)
    print(f"Model saved at: {torch_model_path}")


if __name__ == "__main__":
    main()