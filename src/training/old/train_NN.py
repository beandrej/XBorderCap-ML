import sys; sys.dont_write_bytecode = True
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.train_utils import (
    prepareModel,
    prepareData,
    preparePaths,
    setSeed
)

def main(dataset, model_name, border):

    """
    ************************************************************************************************************************************************************************************
                                                                            PRE-PROCESSING
    ************************************************************************************************************************************************************************************
    """

    setSeed()

    print(f"Using device: {config.DEVICE}")
    
    data_path, model_path, train_metrics_path, test_metrics_path, pred_path  = preparePaths(dataset, model_name, border)

    print(f"Loading data from: {dataset}")
    data = prepareData(dataset, border)
    if data[0] is None:
        return
    X_train, Y_train, X_val, Y_val, pca = data

    model, train_loader, val_loader = prepareModel(model_name, X_train, Y_train, X_val, Y_val)

    criterion = config.REG_CRITERIA()
    optimizer = config.OPTIMIZER(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    train_r2_scores = []
    train_mae_scores = []
    val_r2_scores = []
    val_mae_scores = []
    best_val_r2 = -np.inf 

    if config.USE_PCA:
        print(f"\nPCA ENABLED: Reduction to {config.PCA_COMP} dim. before feeding into Train Loop")
        print(f"PCA {config.PCA_COMP} dims → Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")
    else:
        print("\nPCA DISABLED")

    if model_name == 'LSTM':
        print(f"SEQ_LEN = {config.SEQ_LEN}")

    print(f"\nStarting Hybrid Training on {border} — [{dataset}] | Model: {model_name}")
    print(f"Train/Val Split: {config.TRAIN_SPLIT:.0%}/{config.VALID_SPLIT:.0%}")
    print(f"Batch size: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE} | WD: {config.WEIGHT_DECAY}\n")

    """
    ************************************************************************************************************************************************************************************
                                                                        TRAINING LOOP
    ************************************************************************************************************************************************************************************
    """

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0  

        y_train_true_list = []
        y_train_pred_list = []
        
        for batch_idx, batch in enumerate(train_loader):
            if model_name == "LSTM":
                X_batch, y_batch, lengths = batch
                X_batch = X_batch.to(config.DEVICE)
                y_batch = y_batch.to(config.DEVICE)
                lengths = lengths.to(config.DEVICE)

                y_pred = model(X_batch, lengths)


            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(config.DEVICE)
                y_batch = y_batch.to(config.DEVICE)
                predictions = model(X_batch)

            optimizer.zero_grad()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            y_train_pred = predictions.detach().cpu().numpy()
            y_train_true = y_batch.detach().cpu().numpy()

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
            for batch in val_loader:
                if model_name == "LSTM":
                    X_batch, y_batch, lengths = batch
                    X_batch = X_batch.to(config.DEVICE)
                    y_batch = y_batch.to(config.DEVICE)
                    lengths = lengths.to(config.DEVICE)
                    y_pred = model(X_batch, lengths)

                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(config.DEVICE)
                    y_batch = y_batch.to(config.DEVICE)
                    y_pred = model(X_batch)


                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

                y_pred = y_pred.detach().cpu().numpy()
                y_true = y_batch.detach().cpu().numpy()

                y_val_pred_list.append(y_pred)
                y_val_true_list.append(y_true)    


        y_pred_full = np.vstack(y_val_pred_list)
        y_true_full = np.vstack(y_val_true_list)

        val_r2 = r2_score(y_true_full, y_pred_full)
        val_mae = mean_absolute_error(y_true_full, y_pred_full)

        if val_r2 > best_val_r2 and epoch >= config.MIN_EPOCHS_MODEL_SAVE:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), torch_model_path)

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_r2_scores.append(val_r2)
        val_mae_scores.append(val_mae)

        scheduler.step(val_loss)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.EPOCHS} | Train {criterion.__class__.__name__}: {train_losses[-1]:.2f} | Val {criterion.__class__.__name__}: {val_losses[-1]:.2f} | Train-R²: {train_r2:.2f} | Val-R²: {val_r2:.2f} | Train-MAE: {train_mae:.2f} | Val-MAE: {val_mae:.2f}")


    """
    ************************************************************************************************************************************************************************************
                                                                        METRICS
    ************************************************************************************************************************************************************************************
    """


    epochs_list = list(range(1, config.EPOCHS + 1))
    metrics_df = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_r2': train_r2_scores,
        'val_r2' : val_r2_scores,
        'train_mae': train_mae_scores,
        'val_mae' : val_mae_scores
        
    })

    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
