import os
import sys; sys.dont_write_bytecode = True
import torch
import json
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config
from model import Hybrid, TCN
from utils.train_utils import (
    prepareDataHybrid,
    createDataloadersHybrid,
    createTCNDataloaders,
    preparePaths,
    computeGlobalMAE,
    setSeed
)


# TODO decode cls before computing MAE


def main(dataset, model_name, border):

    """
    ************************************************************************************************************************************************************************************
                                                                            PRE-PROCESSING
    ************************************************************************************************************************************************************************************
    """

    setSeed()
    print(f"Using device: {config.DEVICE}")

    model_path, metrics_path = preparePaths(dataset, model_name, border)

    X_train, Y_cls_train, Y_reg_train, X_val, Y_cls_val, Y_reg_val, label_encs, cls_cols, reg_cols, pca = prepareDataHybrid(dataset, border)

    if border in config.CLS_COLS:
        out_dim = len(label_encs[border])
        print(f"No. of classes: {out_dim}")
        task_type = 'classification'
        criterion = config.CLS_CRITERIA()
    else:
        out_dim = 1
        task_type = 'regression'
        criterion = config.REG_CRITERIA()

    if model_name == 'Hybrid':
        train_loader, val_loader, input_dim = createDataloadersHybrid(
            X_train, Y_cls_train, Y_reg_train,
            X_val, Y_cls_val, Y_reg_val,
            batch_size=config.BATCH_SIZE
        )
        model = Hybrid(input_dim, out_dim, task_type).to(config.DEVICE)

    elif model_name == 'TCN':
        train_loader, val_loader, input_dim = createTCNDataloaders(
            X_train, Y_cls_train, Y_reg_train,
            X_val, Y_cls_val, Y_reg_val,
            batch_size=config.BATCH_SIZE,
            seq_len=config.SEQ_LEN
        )
        model = TCN(input_dim, out_dim, task_type).to(config.DEVICE)

    print(f"\nTarget column: {border} ({task_type})")
    print(f"Train/Val Split: {config.TRAIN_SPLIT:.0%}/{config.VALID_SPLIT:.0%}")
    print(f"Batch size: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE} | WD: {config.WEIGHT_DECAY}")

    if config.USE_PCA:
        print(f"PCA {config.PCA_COMP} dims → Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")

    optimizer = config.OPTIMIZER(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

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

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0

        all_preds = []
        all_targets = []

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(config.DEVICE)
            Y_batch = Y_batch.to(config.DEVICE)

            if task_type == "regression":
                Y_batch = Y_batch.float().view(-1, 1)  # match model output shape
            else:
                Y_batch = Y_batch.long()  # for CrossEntropyLoss or similar


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

        if model.task_type == 'classification':
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

            train_r2_scores.append(0.0)
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
            train_acc_scores.append(0.0)


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

        if model.task_type == 'classification':
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
            val_r2_scores.append(0.0)
        else:
            val_r2 = r2_score(y_val_true_full, y_val_pred_full)
            val_mae = np.mean(np.abs(y_val_true_full - y_val_pred_full))
            val_r2_scores.append(val_r2)
            val_mae_scores.append(val_mae)
            val_acc_scores.append(0.0)

        scheduler.step(val_loss)

        """
        ************************************************************************************************************************************************************************************
                                                                            PRINTS
        ************************************************************************************************************************************************************************************
        """

        if (epoch + 1) % 2 == 0:
            log = f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | "

            if model.task_type == 'regression':
                train_r2 = train_r2_scores[-1]
                val_r2 = val_r2_scores[-1]
                train_mae = train_mae_scores[-1]
                val_mae = val_mae_scores[-1]

                log += f"Train R²: {train_r2:.2f} | Val R²: {val_r2:.2f} | "
                log += f"Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | "

            elif model.task_type == 'classification':
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

            print(log.strip(" | "))

    """
    ************************************************************************************************************************************************************************************
                                                                        METRICS
    ************************************************************************************************************************************************************************************
    """

    model_config = {
        "input_dim": input_dim,
        "cls_cols": cls_cols,
        "reg_cols": reg_cols,
        "hidden_dim": config.HIDDEN_DIM
    }
    model_config_path = os.path.join(model_config_path)
    with open(model_config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    print("Saved border json config")

    metrics_df = pd.DataFrame({
        'epoch': list(range(1, config.EPOCHS + 1)),
        'train_r2': train_r2_scores,
        'val_r2': val_r2_scores,
        'train_global_mae': train_mae_scores,
        'val_global_mae': val_mae_scores,
        'train_cls_acc': train_acc_scores,
        'val_cls_acc': val_acc_scores,
    })

    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}") 


        
        