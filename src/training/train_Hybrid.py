import os
import sys; sys.dont_write_bytecode = True
import torch
import json
import importlib
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config
importlib.reload(config)
from model import getModel
from utils.train_utils import *


def main(dataset, model_name, border):

    """
    ************************************************************************************************************************************************************************************
                                                                            PRE-PROCESSING
    ************************************************************************************************************************************************************************************
    """
    # print("\n[DEBUG] Effective config values:")
    # for attr in ['SHARED_HIDDEN', 'REG_HIDDEN', 'DROPOUT', 'BATCH_SIZE', 'KERNEL_SIZE', 'LEARNING_RATE', 'SEQ_LEN']:
    #     print(f"{attr:15}: {getattr(config, attr)}")
    # if config.KEEP_BORDER:
    #     border = config.BORDER_TO_KEEP
    #     print(f"KEEPING SAME BORDER FOR WHOLE TRAINING: {border}")

    # print(f"\n ----------------------------- {model_name} on {dataset} | Target: {target_col} | Border Type: {dataset.split('_')[1]} ----------------------------- ")
    print(f"\n === Grid Search for {model_name} on {dataset} | Target: {border} === ")
    setSeed()
    border = 'GER_FRA'
    data_path, model_path, train_metrics_path, test_metrics_path, pred_path  = preparePaths(dataset, model_name, border)

    if not os.path.exists(data_path):
        print(f"Data for border '{border}' not found. Skipping...")
        return 

    data = torch.load(data_path, weights_only=False)

    test_timestamps = data["test_timestamp"]
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    task_type = data["task_type"]
    input_dim = data["input_dim"]
    out_dim = data["out_dim"]

    model, task_type, out_dim = getModel(model_name, input_dim, out_dim, task_type)

    train_loader, val_loader = getLoadersHybrid(X_train, Y_train, X_val, Y_val, task_type, model)

    
    if config.TRAIN:
        print("\nTraining Setup Overview")
        print("\n--- Confirmed Config in train_Hybrid ---")

        print(f"Target column      : {border} ({task_type})")
        print(f"Train/Val Split    : {config.TRAIN_SPLIT:.0%}/{config.VALID_SPLIT:.0%}")
        print(f"Batch size         : {config.BATCH_SIZE}")
        print(f"Learning rate      : {config.LEARNING_RATE}")
        print(f"Weight decay       : {config.WEIGHT_DECAY}")
        print(f"Input features     : {input_dim}")
        print(f"Output dimension   : {out_dim}")
        print(f"Using dropout      : {config.DROPOUT}")

        print("\nDataset Sizes")
        print(f"Train set          : {len(X_train):>6} samples")
        print(f"Validation set     : {len(X_val):>6} samples")
        print(f"Test set           : {len(X_test):>6} samples")

        model.to(config.DEVICE)
        optimizer = config.OPTIMIZER(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = config.CLS_CRITERIA() if task_type == 'classification' else config.REG_CRITERIA()

        if config.ENABLE_BACKTEST:
            print(" \n---------------  BACKTESTING MODE --------------- ")
            print(" ------ Testing Range: 2023-07-01 to 2023-12-31 ------ \n")
        
        train_loss_scores = []
        train_r2_scores = []
        train_acc_scores = []
        train_f1_scores = []
        train_prec_scores = []
        train_rec_scores = []
        train_mae_scores = []

        val_loss_scores = [] 
        val_r2_scores = []   
        val_acc_scores = []
        val_f1_scores = []
        val_prec_scores = []
        val_rec_scores = []   
        val_mae_scores = []

        best_val_loss = np.inf

    """
    ************************************************************************************************************************************************************************************
                                                                        TRAINING LOOP
    ************************************************************************************************************************************************************************************
    """
    if config.TRAIN:

        print(f"Using device: {config.DEVICE}")
        
        for epoch in tqdm(range(config.EPOCHS), desc="Training Epochs", leave=False):
            model.train()
            train_loss = 0

            all_preds = []
            all_targets = []

            for X_batch, Y_batch in tqdm(train_loader, desc="Batches", leave=False):
                X_batch = X_batch.to(config.DEVICE, non_blocking=True)
                Y_batch = Y_batch.to(config.DEVICE, non_blocking=True)

                if task_type == "regression":
                    Y_batch = Y_batch.float().view(-1, 1)
                else:
                    Y_batch = Y_batch.long()


                optimizer.zero_grad()

                y_pred = model(X_batch)
                loss = criterion(y_pred, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                all_preds.append(y_pred.detach().cpu())
                all_targets.append(Y_batch.detach().cpu())

            y_pred_full = torch.cat(all_preds, dim=0).numpy()
            y_true_full = torch.cat(all_targets, dim=0).numpy()

            train_loss_scores.append(train_loss)

            if task_type == 'classification':
                pred_labels = y_pred_full.argmax(axis=1)
                true_labels = y_true_full.astype(int)
                train_acc = accuracy_score(true_labels, pred_labels)
                train_prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
                train_rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
                train_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

                with open(os.path.join(config.PROJECT_ROOT, "mappings", f"clsMap.json")) as f:
                    class_map = json.load(f)

                inverse_map = {v: float(k) for k, v in class_map[border].items()}
                decoded_preds = np.array([inverse_map[int(p)] for p in pred_labels])
                decoded_trues = np.array([inverse_map[int(t)] for t in true_labels])
                train_mae = np.mean(np.abs(decoded_preds - decoded_trues))

                train_mae_scores.append(train_mae)
                train_acc_scores.append(train_acc)
                train_prec_scores.append(train_prec)
                train_rec_scores.append(train_rec)
                train_f1_scores.append(train_f1)
            else:
                # Regression metrics
                train_r2 = r2_score(y_true_full, y_pred_full)
                train_mae = np.mean(np.abs(y_true_full - y_pred_full))

                train_r2_scores.append(train_r2)
                train_mae_scores.append(train_mae)


            """
            ************************************************************************************************************************************************************************************
                                                                                VALIDATION
            ************************************************************************************************************************************************************************************
            """


            model.eval()
            val_loss = 0.0
            y_val_pred_list = []
            y_val_true_list = []

            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.to(config.DEVICE)
                    Y_batch = Y_batch.to(config.DEVICE)

                    if task_type == "regression":
                        Y_batch = Y_batch.float().view(-1, 1)  # match model output shape
                    else:
                        Y_batch = Y_batch.long()  # for CrossEntropyLoss or similar

                    y_pred = model(X_batch)
                    loss = criterion(y_pred, Y_batch)
                    val_loss += loss.item()

                    y_val_pred_list.append(y_pred.detach().cpu())
                    y_val_true_list.append(Y_batch.detach().cpu())


            y_val_pred_full = torch.cat(y_val_pred_list, dim=0).numpy()
            y_val_true_full = torch.cat(y_val_true_list, dim=0).numpy()

            val_loss_scores.append(val_loss)

            if val_loss < best_val_loss and epoch >= config.MIN_EPOCHS_MODEL_SAVE:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)

            if task_type == 'classification':
                pred_labels = y_val_pred_full.argmax(axis=1)
                true_labels = y_val_true_full.astype(int)

                val_acc = accuracy_score(true_labels, pred_labels)
                val_prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
                val_rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
                val_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

                with open(os.path.join(config.PROJECT_ROOT, "mappings", f"clsMap.json")) as f:
                    class_map = json.load(f)
                inverse_map = {v: float(k) for k, v in class_map[border].items()}

                decoded_preds = np.array([inverse_map[int(p)] for p in pred_labels])
                decoded_trues = np.array([inverse_map[int(t)] for t in true_labels])
                val_mae = np.mean(np.abs(decoded_preds - decoded_trues))

                val_mae_scores.append(val_mae)
                val_acc_scores.append(val_acc)
                val_prec_scores.append(val_prec)
                val_rec_scores.append(val_rec)
                val_f1_scores.append(val_f1)
            else:
                val_r2 = r2_score(y_val_true_full, y_val_pred_full)
                val_mae = np.mean(np.abs(y_val_true_full - y_val_pred_full))
                val_r2_scores.append(val_r2)
                val_mae_scores.append(val_mae)

            scheduler.step(val_loss)

            """
            ************************************************************************************************************************************************************************************
                                                                                PRINTS
            ************************************************************************************************************************************************************************************
            """

            if (epoch + 1) % 2 == 0:
                log = f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | "

                if task_type == 'regression':
                    train_r2 = train_r2_scores[-1]
                    val_r2 = val_r2_scores[-1]
                    train_mae = train_mae_scores[-1]
                    val_mae = val_mae_scores[-1]

                    log += f"Train R²: {train_r2:.2f} | Val R²: {val_r2:.2f} | "
                    log += f"Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | "

                elif task_type == 'classification':
                    train_acc = train_acc_scores[-1]
                    val_acc = val_acc_scores[-1]
                    train_prec = train_prec_scores[-1]
                    val_prec = val_prec_scores[-1]
                    train_rec = train_rec_scores[-1]
                    val_rec = val_rec_scores[-1]
                    train_f1 = train_f1_scores[-1]
                    val_f1 = val_f1_scores[-1]
                    train_mae = train_mae_scores[-1]
                    val_mae = val_mae_scores[-1]

                    log += f"Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | "
                    log += f"Train Prec: {train_prec:.2f} | Val Prec: {val_prec:.2f} | "
                    log += f"Train Rec: {train_rec:.2f} | Val Rec: {val_rec:.2f} | "
                    log += f"Train F1: {train_f1:.2f} | Val F1: {val_f1:.2f} | "
                    log += f"Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | "

                tqdm.write(log.strip(" | "))


    """
    ************************************************************************************************************************************************************************************
                                                                        FINAL TEST PREDICTION
    ************************************************************************************************************************************************************************************
    """
    if config.PREDICT:

        print(f"\n--------------- TESTING on {border} -----------------\n")

        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        test_loader = getLoadersHybrid(X_test, Y_test, X_test, Y_test, task_type, model)[0]

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(config.DEVICE)
                Y_batch = Y_batch.to(config.DEVICE)

                if task_type == 'regression':
                    Y_batch = Y_batch.float().view(-1, 1)
                else:
                    Y_batch = Y_batch.long()

                y_pred = model(X_batch)

                all_preds.append(y_pred.cpu())
                all_trues.append(Y_batch.cpu())

        preds = torch.cat(all_preds, dim=0).numpy()
        trues = torch.cat(all_trues, dim=0).numpy()

        if task_type == 'regression':
            r2 = r2_score(trues, preds)
            mae = np.mean(np.abs(trues - preds))
            print(f"\nTest R²: {r2:.4f} | MAE: {mae:.4f}")
            test_metrics = {
                'border': border,
                'test_r2': r2,
                'test_mae': mae,
                'test_acc': 0.0,
                'test_prec': 0.0,
                'test_rec': 0.0,
                'test_f1': 0.0
            }

        else:
            pred_labels = preds.argmax(axis=1)
            true_labels = trues.astype(int)

            acc = accuracy_score(true_labels, pred_labels)
            prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
            rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

            with open(os.path.join(config.PROJECT_ROOT, "mappings", f"clsMap.json")) as f:
                class_map = json.load(f)
            inverse_map = {v: float(k) for k, v in class_map[border].items()}

            preds = np.array([inverse_map[int(p)] for p in pred_labels])
            trues = np.array([inverse_map[int(t)] for t in true_labels])
            mae = np.mean(np.abs(preds - trues))

            print(f"\nFinal Test Accuracy: {acc:.4f} | MAE: {mae:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            test_metrics = {
                'border': border,
                'test_r2': 0.0,
                'test_mae': mae,
                'test_acc': acc,
                'test_prec': prec,
                'test_rec': rec,
                'test_f1': f1
            }

    """
    ************************************************************************************************************************************************************************************
                                                                        METRICS
    ************************************************************************************************************************************************************************************
    """

    if config.WRITE_METRICS:

        if config.TRAIN:

            if task_type == "classification":
                train_r2_scores = [0.0] * config.EPOCHS
                val_r2_scores = [0.0] * config.EPOCHS
                train_prec_scores = [0.0] * config.EPOCHS
                val_prec_scores = [0.0] * config.EPOCHS
                train_rec_scores = [0.0] * config.EPOCHS
                val_rec_scores = [0.0] * config.EPOCHS
                train_f1_scores = [0.0] * config.EPOCHS
                val_f1_scores = [0.0] * config.EPOCHS

            elif task_type == "regression":
                train_acc_scores = [0.0] * config.EPOCHS
                val_acc_scores = [0.0] * config.EPOCHS
                train_prec_scores = [0.0] * config.EPOCHS
                val_prec_scores = [0.0] * config.EPOCHS
                train_rec_scores = [0.0] * config.EPOCHS
                val_rec_scores = [0.0] * config.EPOCHS
                train_f1_scores = [0.0] * config.EPOCHS
                val_f1_scores = [0.0] * config.EPOCHS

            metrics_df = pd.DataFrame({
                'epoch': list(range(1, config.EPOCHS + 1)),
                'train_r2': train_r2_scores,
                'val_r2': val_r2_scores,
                'train_mae': train_mae_scores,
                'val_mae': val_mae_scores,
                'train_cls_acc': train_acc_scores,
                'val_cls_acc': val_acc_scores,
                'train_precision': train_prec_scores,
                'val_precision': val_prec_scores,
                'train_recall': train_rec_scores,
                'val_recall': val_rec_scores,
                'train_f1': train_f1_scores,
                'val_f1': val_f1_scores
            })

            metrics_df.to_csv(train_metrics_path, index=False)
            print(f"Training Metrics saved to       : {train_metrics_path}") 
        
        if config.PREDICT:

            if os.path.exists(test_metrics_path):
                df_existing = pd.read_csv(test_metrics_path)
                df_existing = df_existing[df_existing["border"] != border]
            else:
                df_existing = pd.DataFrame()

            test_metrics['border'] = border
            df_updated = pd.concat([df_existing, pd.DataFrame([test_metrics])], ignore_index=True)
            df_updated.to_csv(test_metrics_path, index=False)

            
            print(f"Test Metrics saved to           : {test_metrics_path}")

    if config.WRITE_PREDICTIONS:
        print(f"[DEBUG] preds.shape: {preds.shape}")
        print(f"[DEBUG] trues.shape: {trues.shape}")
        print(f"[DEBUG] timestamps: {len(test_timestamps)}")
        savePredictions(
            pred_path=pred_path,
            test_timestamps=test_timestamps,
            trues=trues,
            preds=preds,
            border=border,
            write=config.WRITE_PREDICTIONS,
            overwrite= config.OVERWRITE_PREDICTIONS,
        )


            