import os
import torch
import json
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from model import HybridOutputMLP
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder


"""
************************************
    CONFIG
************************************
"""
LOOP_TRAINING = False
BORDER_TYPE = "NTC"

DATASET_LOOP = ["BL_FBMC_FULL"]
MODEL_LOOP = ["LSTM", "BaseModel"]
CRITERIA_LOOP = [
    (nn.MSELoss, "MSELoss"),
    (nn.L1Loss, "L1Loss"),
    (nn.SmoothL1Loss, "SmoothL1Loss")
]

TRAINING_SET = "BL_NTC_FULL"
MODEL_NAME = "HYBRID"
CRITERION = nn.L1Loss

TRAIN_SPLIT = 0.9
VALID_SPLIT = 0.15
BATCH_SIZE = 256
EPOCHS = 15
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 3e-4
SEED = 42
SEQ_LEN = 24*7*2

USE_RF_ONLY = False
MAKE_PLOTS = True


"""
************************************
"""

class HybridDataset(torch.utils.data.Dataset):
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

def buildTrainValDataset(dataset):
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{dataset}.csv"), index_col=0)
    print(f"\n USING DATASET: {dataset}\n")

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

    X_val = X_train_full.iloc[val_index:]
    Y_val = Y_train_full.iloc[val_index:]

    print("Shape of X_train:", X_train.shape) 
    print("Shape of Y_train:", Y_train.shape) 
    print("Shape of X_val:", X_val.shape) 
    print("Shape of Y_val:", Y_val.shape) 
    
    return X_train, Y_train, X_val, Y_val

def safe_label_encode(train_col, val_col):
    le = LabelEncoder()
    le.fit(train_col)
    train_encoded = le.transform(train_col)

    class_map = {label: i for i, label in enumerate(le.classes_)}
    val_encoded = val_col.map(class_map).fillna(-1).astype(int)

    return train_encoded, val_encoded, le

def getTargetTypes(Y_train, threshold=100):
    target_types = {}
    for col in Y_train.columns:
        unique_count = Y_train[col].nunique()
        if unique_count > threshold:
            target_types[col] = 'regression'
        else:
            target_types[col] = 'classification'
    return target_types

def main(TRAINING_SET, MODEL_NAME, CRITERION_CLASS):

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

    X_train, Y_train, X_val, Y_val = buildTrainValDataset(TRAINING_SET)

    target_types = getTargetTypes(Y_train)
    cls_cols = [col for col, t in target_types.items() if t == 'classification']
    reg_cols = [col for col, t in target_types.items() if t == 'regression']

    print(f"\nNumber of Regression columns: {len(reg_cols)}")
    print(f"Number of Classification columns: {len(cls_cols)}\n")

    # Encode classification targets
    label_encoders = {}
    Y_cls_train = pd.DataFrame()
    Y_cls_val = pd.DataFrame()

    for col in cls_cols:
        train_encoded, val_encoded, le = safe_label_encode(Y_train[col], Y_val[col])
        Y_cls_train[col] = train_encoded
        Y_cls_val[col] = val_encoded
        label_encoders[col] = le

    # Scale regression targets
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    Y_reg_train = pd.DataFrame()
    Y_reg_val = pd.DataFrame()

    # Scale regression targets
    Y_scaler = StandardScaler()
    Y_reg_train = pd.DataFrame()
    Y_reg_val = pd.DataFrame()

    if reg_cols:
        Y_reg_train = pd.DataFrame(Y_scaler.fit_transform(Y_train[reg_cols]), columns=reg_cols)
        Y_reg_val = pd.DataFrame(Y_scaler.transform(Y_val[reg_cols]), columns=reg_cols)

    # Create torch tensors for training
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)

    X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    X_tensor_val = torch.tensor(X_val, dtype=torch.float32)

    Y_cls_tensors_train = [torch.tensor(Y_cls_train[col].values, dtype=torch.long) for col in Y_cls_train.columns]
    Y_cls_tensors_val = [torch.tensor(Y_cls_val[col].values, dtype=torch.long) for col in Y_cls_val.columns]

    Y_reg_tensors_train = [torch.tensor(Y_reg_train[col].values, dtype=torch.float32).view(-1, 1) for col in Y_reg_train.columns]
    Y_reg_tensors_val = [torch.tensor(Y_reg_val[col].values, dtype=torch.float32).view(-1, 1) for col in Y_reg_val.columns]

    train_dataset = HybridDataset(X_tensor_train, Y_cls_tensors_train, Y_reg_tensors_train)
    val_dataset = HybridDataset(X_tensor_val, Y_cls_tensors_val, Y_reg_tensors_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cls_dims = [len(label_encoders[col].classes_) for col in cls_cols]
    reg_count = len(reg_cols)

    model = HybridOutputMLP(input_dim=X_tensor_train.shape[1], cls_dims=cls_dims, reg_count=reg_count).to(device)

    cls_criterions = [nn.CrossEntropyLoss() for _ in cls_dims]
    reg_criterions = [nn.MSELoss() for _ in range(reg_count)]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    train_r2_scores = []
    train_mae_scores = []
    val_r2_scores = []
    val_mae_scores = []

    train_cls_accs = [[] for _ in range(len(cls_cols))]
    val_cls_accs = [[] for _ in range(len(cls_cols))]
    label_encoders = {}
    class_mappings = {}

    for col in cls_cols:
        le = LabelEncoder()
        Y_train[col] = le.fit_transform(Y_train[col])
        label_encoders[col] = le

        # Safely cast both keys and values to str for JSON
        mapping = {
            str(int(k)): str(v) for k, v in zip(le.transform(le.classes_), le.classes_)
        }
        class_mappings[col] = mapping

    # Save to JSON
    with open("class_mappings.json", "w") as f:
        json.dump(class_mappings, f, indent=2)

    print("✅ Saved class mappings to class_mappings.json")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        y_train_pred_list = []
        y_train_true_list = []

        cls_train_preds = [[] for _ in range(len(cls_cols))]
        cls_train_trues = [[] for _ in range(len(cls_cols))]

        for X_batch, Y_cls_batch, Y_reg_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_cls_batch = [y.to(device) for y in Y_cls_batch]
            Y_reg_batch = [y.to(device) for y in Y_reg_batch]

            optimizer.zero_grad()
            cls_outs, reg_outs = model(X_batch)

            loss = 0
            for i, (out, target) in enumerate(zip(cls_outs, Y_cls_batch)):
                valid_mask = target != -1
                if valid_mask.any():
                    loss += cls_criterions[i](out[valid_mask], target[valid_mask])

                    pred_labels = out[valid_mask].argmax(dim=1).detach().cpu().numpy()
                    true_labels = target[valid_mask].detach().cpu().numpy()
                    cls_train_preds[i].extend(pred_labels)
                    cls_train_trues[i].extend(true_labels)

            for i, (out, target) in enumerate(zip(reg_outs, Y_reg_batch)):
                loss += reg_criterions[i](out, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if len(reg_outs) > 0:
                y_pred = torch.cat([out.detach().cpu() for out in reg_outs], dim=1)
                y_true = torch.cat([target.detach().cpu() for target in Y_reg_batch], dim=1)
                y_pred = Y_scaler.inverse_transform(y_pred.numpy())
                y_true = Y_scaler.inverse_transform(y_true.numpy())

                y_train_pred_list.append(y_pred)
                y_train_true_list.append(y_true)

        if len(reg_outs) > 0:
            y_train_pred_full = np.vstack(y_train_pred_list)
            y_train_true_full = np.vstack(y_train_true_list)

            train_r2 = r2_score(y_train_true_full, y_train_pred_full)
            train_mae = mean_absolute_error(y_train_true_full, y_train_pred_full)

            train_r2_scores.append(train_r2)
            train_mae_scores.append(train_mae)

            per_head_train_r2 = [r2_score(y_train_true_full[:, i], y_train_pred_full[:, i]) for i in range(len(reg_cols))]
            per_head_train_mae = [mean_absolute_error(y_train_true_full[:, i], y_train_pred_full[:, i]) for i in range(len(reg_cols))]
        else:
            train_r2 = train_mae = 0.0
            per_head_train_r2 = []
            per_head_train_mae = []

        per_head_train_accs = []
        for i in range(len(cls_cols)):
            if cls_train_trues[i]:
                acc = accuracy_score(cls_train_trues[i], cls_train_preds[i])
                per_head_train_accs.append(acc)
                train_cls_accs[i].append(acc)
            else:
                per_head_train_accs.append(None)

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0
        y_val_pred_list = []
        y_val_true_list = []

        cls_val_preds = [[] for _ in range(len(cls_cols))]
        cls_val_trues = [[] for _ in range(len(cls_cols))]

        with torch.no_grad():
            for X_batch, Y_cls_batch, Y_reg_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_cls_batch = [y.to(device) for y in Y_cls_batch]
                Y_reg_batch = [y.to(device) for y in Y_reg_batch]

                cls_outs, reg_outs = model(X_batch)

                loss = 0
                for i, (out, target) in enumerate(zip(cls_outs, Y_cls_batch)):
                    valid_mask = target != -1
                    if valid_mask.any():
                        loss += cls_criterions[i](out[valid_mask], target[valid_mask])

                        pred_labels = out[valid_mask].argmax(dim=1).cpu().numpy()
                        true_labels = target[valid_mask].cpu().numpy()
                        cls_val_preds[i].extend(pred_labels)
                        cls_val_trues[i].extend(true_labels)

                for i, (out, target) in enumerate(zip(reg_outs, Y_reg_batch)):
                    loss += reg_criterions[i](out, target)

                val_loss += loss.item()

                if len(reg_outs) > 0:
                    y_pred = torch.cat([out.cpu() for out in reg_outs], dim=1)
                    y_true = torch.cat([target.cpu() for target in Y_reg_batch], dim=1)
                    y_pred = Y_scaler.inverse_transform(y_pred.numpy())
                    y_true = Y_scaler.inverse_transform(y_true.numpy())

                    y_val_pred_list.append(y_pred)
                    y_val_true_list.append(y_true)

        if len(reg_outs) > 0:
            y_val_pred_full = np.vstack(y_val_pred_list)
            y_val_true_full = np.vstack(y_val_true_list)

            val_r2 = r2_score(y_val_true_full, y_val_pred_full)
            val_mae = mean_absolute_error(y_val_true_full, y_val_pred_full)

            val_r2_scores.append([r2_score(y_val_true_full[:, i], y_val_pred_full[:, i]) for i in range(len(reg_cols))])
            val_mae_scores.append([mean_absolute_error(y_val_true_full[:, i], y_val_pred_full[:, i]) for i in range(len(reg_cols))])
        else:
            val_r2 = val_mae = 0.0
            val_r2_scores.append([])
            val_mae_scores.append([])

        per_head_val_accs = []
        for i in range(len(cls_cols)):
            if cls_val_trues[i]:
                acc = accuracy_score(cls_val_trues[i], cls_val_preds[i])
                per_head_val_accs.append(acc)
                val_cls_accs[i].append(acc)
            else:
                per_head_val_accs.append(None)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | "
            f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if reg_cols:
                print("→ Per-head Val R² (Regression):")
                for col, r2 in zip(reg_cols, val_r2_scores[-1]):
                    print(f"   - {col}: {r2:.4f}")

            if cls_cols:
                print("→ Per-head Val Accuracy (Classification):")
                for col, acc in zip(cls_cols, per_head_val_accs):
                    if acc is not None:
                        print(f"   - {col}: {acc:.4f}")
                    else:
                        print(f"   - {col}: skipped (no valid samples)")

    # -------------------- 📊 PLOTS --------------------
    if reg_cols:
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(reg_cols):
            r2s = [r2_epoch[i] for r2_epoch in val_r2_scores if len(r2_epoch) > i]
            plt.plot(r2s, label=col)
        plt.title("Per-Head R² (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("R² Score")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        for i, col in enumerate(reg_cols):
            maes = [mae_epoch[i] for mae_epoch in val_mae_scores if len(mae_epoch) > i]
            plt.plot(maes, label=col)
        plt.title("Per-Head MAE (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Absolute Error")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    if cls_cols:
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(cls_cols):
            accs = val_cls_accs[i]
            plt.plot(accs, label=col)
        plt.title("Per-Head Accuracy (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    # -------------------- 💾 SAVE METRICS & MODEL --------------------
    if reg_cols:
        r2_df = pd.DataFrame(val_r2_scores, columns=[f"{col}_R2" for col in reg_cols])
        mae_df = pd.DataFrame(val_mae_scores, columns=[f"{col}_MAE" for col in reg_cols])
        regression_metrics = pd.concat([r2_df, mae_df], axis=1)
        regression_metrics.insert(0, "Epoch", list(range(1, len(regression_metrics) + 1)))
        regression_metrics.to_csv("regression_metrics.csv", index=False)
        print("✅ Saved regression metrics to regression_metrics.csv")

    if cls_cols:
        acc_data = {f"{col}_Accuracy": val_cls_accs[i] for i, col in enumerate(cls_cols)}
        classification_metrics = pd.DataFrame(acc_data)
        classification_metrics.insert(0, "Epoch", list(range(1, len(classification_metrics) + 1)))
        classification_metrics.to_csv("classification_metrics.csv", index=False)
        print("✅ Saved classification metrics to classification_metrics.csv")

    # Save final model
    torch.save(model.state_dict(), "hybrid_model_final.pth")
    model_config = {
    "input_dim": X_tensor_train.shape[1],
    "cls_dims": cls_dims,
    "reg_count": reg_count,
    "cls_cols": cls_cols,
    "reg_cols": reg_cols
}

    with open("model_config.json", "w") as f:
        json.dump(model_config, f)
    print("✅ Saved model to hybrid_model_final.pth")





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

    summary_data = {
        'ModelType': MODEL_NAME,
        'Dataset': TRAINING_SET,
        'Loss Criterion': criterion.__class__.__name__,
        'Learning Rate': f"{LEARNING_RATE:.6f}",
        'Weight Decay': f"{WEIGHT_DECAY:.6f}",
        'Train Split': TRAIN_SPLIT,
        'Val Split' : VALID_SPLIT,
        'Batch size': BATCH_SIZE,
        'Epochs': EPOCHS,
        'Train loss': round(train_losses[-1], 2),
        'Train loss MIN': round(min(train_losses), 2),
        'Val loss': round(val_losses[-1], 2),
        'Val loss MIN': round(min(val_losses), 2),
        'Train-R2-Score': round(train_r2_scores[-1], 2),
        'Val-R2-Score' : round(val_r2_scores[-1], 2),
        'Train-MAE': round(train_mae_scores[-1], 3),
        'Val-MAE' : round(val_mae_scores[-1], 3)
    }

    if MAKE_PLOTS:
        plotTrainValLoss(metrics_df, criterion.__class__.__name__, TRAINING_SET, MODEL_NAME)
        plotR2(metrics_df, TRAINING_SET, MODEL_NAME, criterion.__class__.__name__)


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

    csv_path_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_metrics/{BORDER_TYPE}/{MODEL_NAME}')
    csv_path = os.path.join(csv_path_base_path, f"metrics_{MODEL_NAME}_{TRAINING_SET}_{criterion.__class__.__name__}.csv")
    os.makedirs(csv_path_base_path, exist_ok=True)
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")

    torch_model_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/model_params/{BORDER_TYPE}/{MODEL_NAME}')
    torch_model_path = os.path.join(torch_model_base_path, f"{MODEL_NAME}_{TRAINING_SET}_{criterion.__class__.__name__}.pth")
    os.makedirs(torch_model_base_path, exist_ok=True)
    torch.save(model.state_dict(), torch_model_path)
    print(f"Model saved at: {torch_model_path}")

"""
if __name__ == "__main__":

    if LOOP_TRAINING:
        for loss, loss_name in CRITERIA_LOOP:
            for model_name in MODEL_LOOP:
                for dataset in DATASET_LOOP:
                    print(f"\n====== Running Loop with {model_name} on {dataset} with {loss_name} ======\n")
                    main(dataset, model_name, loss)
    else:
        main(TRAINING_SET, MODEL_NAME, CRITERION)
        
        