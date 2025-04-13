import os
import sys; sys.dont_write_bytecode = True
import torch
import json
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, accuracy_score


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config
from model import Hybrid
from utils.train_utils import (
    prepareDataHybrid,
    createDataloadersHybrid,
    preparePaths,
    computeGlobalMAE,
    setSeed
)

def main(dataset, model_name, border):

    setSeed()

    print(f"Using device: {config.DEVICE}")

    model_path, model_config_path, _, metrics_path = preparePaths(dataset, model_name, border)

    X_train, Y_cls_train, Y_reg_train, X_val, Y_cls_val, Y_reg_val, label_encs, cls_cols, reg_cols, pca = prepareDataHybrid(dataset, border)
        
    train_loader, val_loader, input_dim = createDataloadersHybrid(
        X_train, Y_cls_train, Y_reg_train,
        X_val, Y_cls_val, Y_reg_val,
        batch_size=config.BATCH_SIZE
    )

    model = Hybrid(input_dim, [len(label_encs[c].classes_) for c in cls_cols], len(reg_cols), config.HIDDEN_DIM).to(config.DEVICE)

    print(f"\nNumber of Regression columns: {len(reg_cols)}")
    print(f"Number of Classification columns: {len(cls_cols)}\n")

    cls_dims = [len(label_encs[col].classes_) for col in cls_cols]
    reg_count = len(reg_cols)

    cls_criterions = [config.CLS_CRITERIA() for _ in cls_cols]
    reg_criterions = [config.REG_CRITERIA() for _ in reg_cols]
    optimizer = config.OPTIMIZER(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    if config.USE_PCA:
        print(f"PCA {config.PCA_COMP} dims → Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")

    print(f"\nStarting Hybrid Training on {border} — [{dataset}] | Model: {model_name}")
    print(f"Train/Val Split: {config.TRAIN_SPLIT:.0%}/{config.VALID_SPLIT:.0%}")
    print(f"Batch size: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE} | WD: {config.WEIGHT_DECAY}\n")
    

    train_loss_scores = []
    val_loss_scores = []
    train_r2_scores = []
    val_r2_scores = []
    train_acc_scores = []
    val_acc_scores = []
    train_global_mae_scores = []
    val_global_mae_scores = []

    train_cls_accs = [[] for _ in range(len(cls_cols))]
    val_cls_accs = [[] for _ in range(len(cls_cols))]

    best_val_loss = np.inf

    """
    ************************************************************************************************************************************************************************************
                                                                        TRAINING LOOP
    ************************************************************************************************************************************************************************************
    """


    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        y_train_pred_list = []
        y_train_true_list = []

        cls_train_preds = [[] for _ in range(len(cls_cols))]
        cls_train_trues = [[] for _ in range(len(cls_cols))]

        for X_batch, Y_cls_batch, Y_reg_batch in train_loader:
            X_batch = X_batch.to(config.DEVICE)
            Y_cls_batch = [y.to(config.DEVICE) for y in Y_cls_batch]
            Y_reg_batch = [y.to(config.DEVICE) for y in Y_reg_batch]

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
            train_loss_scores.append(train_loss)

            if len(reg_outs) > 0:
                y_pred = torch.cat([out.detach().cpu() for out in reg_outs], dim=1)
                y_true = torch.cat([target.detach().cpu() for target in Y_reg_batch], dim=1)

                y_pred = y_pred.numpy()
                y_true = y_true.numpy()

                y_train_pred_list.append(y_pred)
                y_train_true_list.append(y_true)

        if len(reg_outs) > 0:
            y_train_pred_full = np.vstack(y_train_pred_list)
            y_train_true_full = np.vstack(y_train_true_list)

            train_r2 = r2_score(y_train_true_full, y_train_pred_full)
            train_r2_scores.append(train_r2)
        else:
            train_r2 = train_global_mae = 0.0
            train_r2_scores.append(train_r2)
        
        train_r2_mean = train_r2 if isinstance(train_r2, float) else np.mean(train_r2)
        train_global_mae = computeGlobalMAE(
            y_train_pred_full if reg_cols else [],
            y_train_true_full if reg_cols else [],
            cls_train_preds,
            cls_train_trues
        )
        train_global_mae_scores.append(train_global_mae)

        per_head_train_accs = []
        for i in range(len(cls_cols)):
            if cls_train_trues[i]:
                acc = accuracy_score(cls_train_trues[i], cls_train_preds[i])
                per_head_train_accs.append(acc)
                train_cls_accs[i].append(acc)
            else:
                per_head_train_accs.append(0.0)

        valid_train_accs = [acc for acc in per_head_train_accs if acc is not None]
        train_acc_scores.append(np.mean(valid_train_accs) if valid_train_accs else None)



        """
        ************************************************************************************************************************************************************************************
                                                                            VALIDATION
        ************************************************************************************************************************************************************************************
        """


        model.eval()
        val_loss = 0.0
        y_val_pred_list = []
        y_val_true_list = []

        cls_val_preds = [[] for _ in range(len(cls_cols))]
        cls_val_trues = [[] for _ in range(len(cls_cols))]

        with torch.no_grad():
            for X_batch, Y_cls_batch, Y_reg_batch in val_loader:
                X_batch = X_batch.to(config.DEVICE)
                Y_cls_batch = [y.to(config.DEVICE) for y in Y_cls_batch]
                Y_reg_batch = [y.to(config.DEVICE) for y in Y_reg_batch]

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

                val_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                val_loss_scores.append(val_loss)

                if val_loss < best_val_loss and epoch >= config.MIN_EPOCHS_MODEL_SAVE:
                    best_val_loss = val_loss
                    model_path = os.path.join(model_path)
                    torch.save(model.state_dict(), model_path)

                if len(reg_outs) > 0:
                    y_pred = torch.cat([out.cpu() for out in reg_outs], dim=1)
                    y_true = torch.cat([target.cpu() for target in Y_reg_batch], dim=1)

                    y_pred = y_pred.numpy()
                    y_true = y_true.numpy()

                    y_val_pred_list.append(y_pred)
                    y_val_true_list.append(y_true)

        if len(reg_outs) > 0:
            y_val_pred_full = np.vstack(y_val_pred_list)
            y_val_true_full = np.vstack(y_val_true_list)

            val_r2 = r2_score(y_val_true_full, y_val_pred_full)
            val_r2_scores.append(val_r2)
        else:
            val_r2_mean = 0.0
            val_r2_scores.append(val_r2_mean)

        val_global_mae = computeGlobalMAE(
            y_val_pred_full if reg_cols else [],
            y_val_true_full if reg_cols else [],
            cls_val_preds,
            cls_val_trues
        )
        val_global_mae_scores.append(val_global_mae)

        per_head_val_accs = []
        for i in range(len(cls_cols)):
            if cls_val_trues[i]:
                acc = accuracy_score(cls_val_trues[i], cls_val_preds[i])
                per_head_val_accs.append(acc)
                val_cls_accs[i].append(acc)
            else:
                per_head_val_accs.append(None)

        valid_accs = [acc for acc in per_head_val_accs if acc is not None]
        val_acc_scores.append(np.mean(valid_accs) if valid_accs else None)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            log = f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "

            # Add regression metrics if present
            if reg_cols:
                train_r2_mean = train_r2 if isinstance(train_r2, float) else np.mean(train_r2)
                val_r2_mean = np.mean(val_r2_scores[-1]) if isinstance(val_r2_scores[-1], list) else val_r2_scores[-1]
                log += f"Train Reg R²: {train_r2_mean:.4f} | Val Reg R²: {val_r2_mean:.4f} | "
                log += f"Train MAE: {train_global_mae:.4f} | Val MAE: {val_global_mae:.4f} | "

            # Add classification metrics if present
            if cls_cols:
                train_acc = f"{train_acc_scores[-1]:.4f}" if train_acc_scores[-1] is not None else "N/A"
                val_acc = f"{val_acc_scores[-1]:.4f}" if val_acc_scores[-1] is not None else "N/A"
                log += f"Train Cls Acc.: {train_acc} | Val Cls Acc.: {val_acc} | "
                log += f"Train MAE: {train_global_mae:.4f} | Val MAE: {val_global_mae:.4f} | "

            print(log.strip(" | "))

    """
    ************************************************************************************************************************************************************************************
                                                                        METRICS
    ************************************************************************************************************************************************************************************
    """

    model_config = {
        "input_dim": input_dim,
        "cls_dims": cls_dims,
        "reg_count": reg_count,
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
        'train_global_mae': train_global_mae_scores,
        'val_global_mae': val_global_mae_scores,
        'train_cls_acc': train_acc_scores,
        'val_cls_acc': val_acc_scores,
    })

    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}") 


        
        