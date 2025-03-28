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

BORDER_TYPE = "MAXBEX"
TRAINING_SET = "BL_FBMC_TIME"
MODEL_NAME = "BaseModel"


TRAIN_SPLIT = 0.9
VALID_SPLIT = 0.3
BATCH_SIZE = 256
EPOCHS = 100
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 0.0001


"""
************************************
"""

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{TRAINING_SET}.csv"), index_col=0)
    

    # Dataset X & Y has to merged (only use intersecting timestamps), they are separated again here..
    if BORDER_TYPE == "MAXBEX":
        first_target_idx = full_df.columns.get_loc("AUS_CZE")
    elif BORDER_TYPE == "NTC":
        first_target_idx = full_df.columns.get_loc("AT_to_IT_NORD")
    else:
        raise ValueError("Wrong BORDER_TYPE!")


    X = full_df.iloc[:, :first_target_idx]
    Y = full_df.iloc[:, first_target_idx:]

    split_index = int(len(full_df) * TRAIN_SPLIT)
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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print("Model parameters:\n", f"Train split: {TRAIN_SPLIT}\n", f"Validation split: {VALID_SPLIT}\n",
        f"Batch size: {BATCH_SIZE}\n", f"Learning rate: {LEARNING_RATE}")

    train_losses = []
    val_losses = []
    train_r2_scores = []
    train_mae_scores = []
    val_r2_scores = []
    val_mae_scores = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0  
        y_train_true_list = []
        y_train_pred_list = []
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

            y_train_pred = predictions.detach().cpu().numpy()
            y_train_true = y_batch.detach().cpu().numpy()

            y_train_pred = Y_scaler.inverse_transform(y_train_pred)
            y_train_true = Y_scaler.inverse_transform(y_train_true)

            y_train_pred_list.append(y_train_pred)
            y_train_true_list.append(y_train_true)

            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
        
        y_train_pred_full = np.vstack(y_train_pred_list)
        y_train_true_full = np.vstack(y_train_true_list)

        train_r2 = r2_score(y_train_true_full, y_train_pred_full)
        train_mae = mean_absolute_error(y_train_true_full, y_train_pred_full)

        train_r2_scores.append(train_r2)
        train_mae_scores.append(train_mae)

        model.eval()
        val_loss = 0
        y_val_true_list = []
        y_val_pred_list = []
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

                y_val_pred_list.append(y_pred)
                y_val_true_list.append(y_true)    

        y_pred_full = np.vstack(y_val_pred_list)
        y_true_full = np.vstack(y_val_true_list)

        val_r2 = r2_score(y_true_full, y_pred_full)
        val_mae = mean_absolute_error(y_true_full, y_pred_full)

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_r2_scores.append(val_r2)
        val_mae_scores.append(val_mae)

        if epoch % 2 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss (MSE): {train_losses[-1]:.2f} | Val Loss (MSE): {val_losses[-1]:.2f}, Train-R²: {train_r2:.2f}, Val-R²: {val_r2:.2f}, Train-MAE: {train_mae:.2f}, Val-MAE: {val_mae:.2f}")
        scheduler.step()

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
        'Train-R2-Score': round(train_r2_scores[-1], 2),
        'Val-R2-Score' : round(val_r2_scores[-1], 2),
        'Train-MAE': round(train_mae_scores[-1], 3),
        'Val-MAE' : round(val_mae_scores[-1], 3)
    }

    summary_df = pd.DataFrame([summary_data])

    summaryXLSX_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_metrics', "00_summary.xlsx")

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